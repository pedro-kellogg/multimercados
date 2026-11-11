"""
Does a given Brazilian hedge fund (or set of funds, by CNPJ) improve a
diversified BRL portfolio's Sharpe ratio?

Run:  pip install -r requirements.txt
      streamlit run hf_sharpe_app.py

Data
- CVM open data: daily fund quotas/AUM (informe diário) + fund registry (cad_fi)
- BCB SGS: CDI (monthly), IMA-B, PTAX USD/BRL
- Yahoo Finance: Ibovespa, MSCI World ETF (USD -> BRL via PTAX)
- NEFIN (optional): Brazilian equity factors

Downloads are cached in ./data_cache, so only the first run is slow.
"""

import io
import os
import sys
import threading
import time
import re
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st
import statsmodels.api as sm
import altair as alt
from scipy.stats import norm

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
os.makedirs(CACHE, exist_ok=True)

CVM_DAILY = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/"
CVM_CAD = "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv"
# Post-CVM 175 registry (funds/classes registered or migrated since 2023)
CVM_CAD_NEW = "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/registro_fundo_classe.zip"
BCB_SGS = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
NEFIN = "https://nefin.com.br/resources/risk_factors/{name}_Factor.xls"
HEADERS = {"User-Agent": "Mozilla/5.0 (research script)"}


# ----------------------------------------------------------------------------
# Download helpers
# ----------------------------------------------------------------------------
def _get(url, timeout=300):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.content


def _cached_bytes(url, fname, refresh=False):
    path = os.path.join(CACHE, fname)
    if os.path.exists(path) and not refresh:
        with open(path, "rb") as f:
            return f.read()
    content = _get(url)
    with open(path, "wb") as f:
        f.write(content)
    return content


# ----------------------------------------------------------------------------
# CVM
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_registry():
    raw = _cached_bytes(CVM_CAD, "cad_fi.csv")
    df = pd.read_csv(io.BytesIO(raw), sep=";", encoding="latin-1",
                     dtype=str, low_memory=False)
    keep = ["CNPJ_FUNDO", "DENOM_SOCIAL", "CLASSE", "FUNDO_COTAS",
            "FUNDO_EXCLUSIVO", "CPF_CNPJ_GESTOR", "GESTOR", "DT_CANCEL",
            "INVEST_QUALIF", "DT_REG", "DT_CONST", "DT_INI_ATIV"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["CNPJ_FUNDO"] = df["CNPJ_FUNDO"].str.replace(r"\D", "", regex=True)
    df = df.dropna(subset=["CNPJ_FUNDO"]).drop_duplicates("CNPJ_FUNDO", keep="last")
    return df


@st.cache_data(show_spinner=False)
def load_registry_new():
    """Names (and manager, if present) from the CVM 175 registry, which covers
    funds missing from the legacy cad_fi.csv. Column names are matched
    case-insensitively because CVM has changed them over time."""
    raw = _cached_bytes(CVM_CAD_NEW, "registro_fundo_classe.zip")
    out = []
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with z.open(name) as f:
                df = pd.read_csv(f, sep=";", encoding="latin-1", dtype=str,
                                 low_memory=False)
            low = {c.lower(): c for c in df.columns}
            name_col = low.get("denominacao_social")
            if not name_col:
                continue
            gest_col = low.get("gestor")
            for key in ("cnpj_classe", "cnpj_fundo", "cnpj_subclasse"):
                if key in low:
                    part = pd.DataFrame({
                        "CNPJ_FUNDO": df[low[key]].str.replace(r"\D", "", regex=True),
                        "DENOM_SOCIAL": df[name_col],
                        "GESTOR": df[gest_col] if gest_col else np.nan})
                    out.append(part)
    if not out:
        return pd.DataFrame(columns=["CNPJ_FUNDO", "DENOM_SOCIAL", "GESTOR"])
    new = pd.concat(out).dropna(subset=["CNPJ_FUNDO", "DENOM_SOCIAL"])
    return new.drop_duplicates("CNPJ_FUNDO", keep="last")


def enrich_registry(reg, cnpjs, manual_names):
    """Fill names for selected funds missing from cad_fi.csv (CVM 175
    registry first, then any name typed next to the CNPJ)."""
    reg = reg.copy()
    have = reg.set_index("CNPJ_FUNDO")["DENOM_SOCIAL"]
    missing = [c for c in cnpjs if pd.isna(have.get(c))]
    if missing:
        try:
            new = load_registry_new()
            add = new[new["CNPJ_FUNDO"].isin(missing)]
            reg = pd.concat([reg[~reg["CNPJ_FUNDO"].isin(add["CNPJ_FUNDO"])
                                 | reg["DENOM_SOCIAL"].notna()], add],
                            ignore_index=True)
            reg = reg.drop_duplicates("CNPJ_FUNDO", keep="first")
        except Exception as e:
            st.warning(f"Couldn't load the CVM 175 registry for names: {e}")
    for c, n in manual_names.items():
        if c in set(reg["CNPJ_FUNDO"]):
            reg.loc[reg["CNPJ_FUNDO"] == c, "DENOM_SOCIAL"] = n
        else:
            reg = pd.concat([reg, pd.DataFrame({"CNPJ_FUNDO": [c],
                                                "DENOM_SOCIAL": [n]})],
                            ignore_index=True)
    return reg


_ABBR = [
    (r"FUNDOS? DE INVESTIMENTO EM COTAS DE FUNDOS DE INVESTIMENTO", "FIC FI"),
    (r"FUNDO DE INVESTIMENTO EM COTAS DE FUNDOS? DE INVESTIMENTO", "FIC FI"),
    (r"FUNDO DE INVESTIMENTO EM COTAS", "FIC"),
    (r"FUNDOS? DE INVESTIMENTO", "FI"),
    (r"MULTIMERCADO", "MM"), (r"CR[ÉE]DITO PRIVADO", "CP"),
    (r"INVESTIMENTO NO EXTERIOR", "IE"), (r"RESPONSABILIDADE LIMITADA", "RL"),
    (r"LONGO PRAZO", "LP"), (r"\bA[ÇC][ÕO]ES\b", "FIA"),
]


def short_name(n, maxlen=45):
    n = re.sub(r"\s+", " ", str(n).upper()).strip()
    for pat, rep_ in _ABBR:
        n = re.sub(pat, rep_, n)
    n = re.sub(r"\s*-\s*", " - ", n).strip(" -")
    return n if len(n) <= maxlen else n[:maxlen - 1].rstrip() + "…"


def fund_date_bounds(reg, cnpjs):
    """Earliest start / latest cancel among the selected funds, from the
    registry, so we only download the years that can contain their data.
    Returns (None, None) if any fund is missing from the registry."""
    r = reg[reg["CNPJ_FUNDO"].isin(cnpjs)]
    if len(r) < len(set(cnpjs)):
        return None, None
    dcols = [c for c in ["DT_INI_ATIV", "DT_CONST", "DT_REG"] if c in r]
    first = None
    if dcols:
        starts = pd.concat([pd.to_datetime(r[c], errors="coerce")
                            for c in dcols], axis=1).min(axis=1)
        first = starts.min() if starts.notna().all() else None
    last = None
    if "DT_CANCEL" in r:
        cancel = pd.to_datetime(r["DT_CANCEL"], errors="coerce")
        last = cancel.max() if cancel.notna().all() else None
    return first, last


def _daily_file_list(start_year, end_date):
    """(url, cache_name, label) for every daily-informe file in the window."""
    files = []
    for y in range(start_year, end_date.year + 1):
        if y <= 2020:
            files.append((f"{CVM_DAILY}HIST/inf_diario_fi_{y}.zip",
                          f"inf_diario_fi_{y}.zip", str(y)))
        else:
            last_m = end_date.month if y == end_date.year else 12
            for m in range(1, last_m + 1):
                ym = f"{y}{m:02d}"
                files.append((f"{CVM_DAILY}inf_diario_fi_{ym}.zip",
                              f"inf_diario_fi_{ym}.zip", ym))
    return files


def _parse_daily_zip(path):
    """Month-end quota, AUM and quotaholders for ALL funds in one CVM file."""
    want = {"DT_COMPTC", "VL_QUOTA", "VL_PATRIM_LIQ", "NR_COTST"}
    out = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with z.open(name) as f:
                header = f.readline().decode("latin-1").strip().split(";")
            cnpj_col = next(c for c in header if c.startswith("CNPJ_FUNDO"))
            cols = [cnpj_col] + [c for c in header if c in want]
            try:  # pyarrow parser: several times faster than the C parser
                with z.open(name) as f:
                    df = pd.read_csv(f, sep=";", encoding="latin-1", usecols=cols,
                                     engine="pyarrow", dtype={cnpj_col: str})
            except Exception:
                with z.open(name) as f:
                    df = pd.read_csv(f, sep=";", encoding="latin-1", usecols=cols,
                                     dtype={cnpj_col: str})
            df = df.rename(columns={cnpj_col: "CNPJ_FUNDO"})
            df["CNPJ_FUNDO"] = df["CNPJ_FUNDO"].str.replace(r"\D", "", regex=True)
            df["DT"] = pd.to_datetime(df["DT_COMPTC"], errors="coerce")
            for c in ["VL_QUOTA", "VL_PATRIM_LIQ", "NR_COTST"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            # Post-175 files can carry subclasses: keep the largest row
            df = (df.sort_values("VL_PATRIM_LIQ")
                    .drop_duplicates(["CNPJ_FUNDO", "DT"], keep="last"))
            df["M"] = df["DT"].dt.to_period("M").astype(str)
            df = df.sort_values("DT").groupby(["CNPJ_FUNDO", "M"]).tail(1)
            out.append(df[["CNPJ_FUNDO", "M", "DT", "VL_QUOTA",
                           "VL_PATRIM_LIQ", "NR_COTST"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


_PARSE_SLOTS = threading.Semaphore(2)


def _download(url, path):
    """Stream to disk (keeps memory low on Streamlit Cloud)."""
    tmp = path + ".part"
    with requests.get(url, headers=HEADERS, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(tmp, path)


def _monthly_file(url, fname, label, keep_zip=False):
    """Download + reduce one CVM file to a small month-end parquet for all
    funds. Cached forever (except the current month, which is still filling)."""
    pq = os.path.join(CACHE, f"me_{label}.parquet")
    current = label == date.today().strftime("%Y%m")
    if os.path.exists(pq) and not current:
        return pd.read_parquet(pq)
    zpath = os.path.join(CACHE, fname)
    if not os.path.exists(zpath) or current:
        _download(url, zpath)
    with _PARSE_SLOTS:  # cap concurrent parses so big yearly files fit in RAM
        part = _parse_daily_zip(zpath)
    part.to_parquet(pq, index=False)
    if not keep_zip:
        os.remove(zpath)  # the raw zip is no longer needed
    return part


def load_fund_panel(start_year, end_date, cnpjs, progress=None, workers=6):
    """Month-end panel for the given CNPJs. Files are fetched in parallel and
    each is reduced once for all funds, so changing CNPJs later is instant."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    files = _daily_file_list(start_year, end_date)
    cn = list(cnpjs)
    frames, failed, done = [], [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_monthly_file, *f): f[2] for f in files}
        for fut in as_completed(futs):
            label = futs[fut]
            done += 1
            if progress is not None:
                progress.progress(done / len(files),
                                  text=f"CVM data: {done}/{len(files)} files")
            try:
                part = fut.result()
            except Exception as e:  # e.g. file not yet published
                failed.append(f"{label} ({type(e).__name__})")
                continue
            if not part.empty:
                frames.append(part[part["CNPJ_FUNDO"].isin(cn)])
    if failed and progress is not None:
        st.warning("Skipped CVM files: " + ", ".join(sorted(failed)))
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    if panel.empty:
        return panel
    panel["M"] = pd.PeriodIndex(panel["M"], freq="M")
    panel = (panel.sort_values("DT")
                  .drop_duplicates(["CNPJ_FUNDO", "M"], keep="last"))
    return panel


# ----------------------------------------------------------------------------
# BCB / Yahoo / NEFIN
# ----------------------------------------------------------------------------
def _sgs_chunk(code, s, e, tries=4):
    """One SGS request with retries. BCB sometimes answers with an empty or
    HTML page (throttling/maintenance) instead of JSON, so retry with backoff."""
    params = {"formato": "json", "dataInicial": s.strftime("%d/%m/%Y"),
              "dataFinal": e.strftime("%d/%m/%Y")}
    last = None
    for k in range(tries):
        try:
            r = requests.get(BCB_SGS.format(code=code), params=params,
                             headers={**HEADERS, "Accept": "application/json"},
                             timeout=60)
            if r.status_code == 404:  # no observations in this range
                return pd.DataFrame()
            r.raise_for_status()
            return pd.DataFrame(r.json())
        except (ValueError, requests.RequestException) as ex:
            last = ex
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"BCB SGS series {code} ({s:%d/%m/%Y}–{e:%d/%m/%Y}) "
                       f"did not return data after {tries} tries: "
                       f"{type(last).__name__}")


@st.cache_data(show_spinner=False)
def sgs(code, start, end):
    """SGS series, fetched in 5-year chunks (API limit is 10y for daily)."""
    frames = []
    s = pd.Timestamp(start)
    end = min(pd.Timestamp(end), pd.Timestamp(date.today()))
    while s <= end:
        e = min(s + pd.DateOffset(years=5) - pd.Timedelta(days=1), end)
        frames.append(_sgs_chunk(code, s, e))
        s = e + pd.Timedelta(days=1)
        time.sleep(0.3)  # be gentle; BCB throttles bursts
    df = pd.concat(frames)
    if df.empty:
        raise RuntimeError(f"BCB SGS series {code} returned no data.")
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    out = df.set_index("data")["valor"].sort_index()
    return out[~out.index.duplicated(keep="last")]


@st.cache_data(show_spinner=False)
def yahoo_monthly_close(ticker, start, end):
    import yfinance as yf
    px = yf.download(ticker, start=start, end=end, auto_adjust=True,
                     progress=False)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    return px.resample("ME").last()


@st.cache_data(show_spinner=False)
def nefin_factors():
    out = {}
    for name in ["Market", "SMB", "HML", "WML", "IML"]:
        try:
            raw = _cached_bytes(NEFIN.format(name=name), f"nefin_{name}.xls")
            df = pd.read_excel(io.BytesIO(raw))
            df.columns = [c.lower() for c in df.columns]
            df["date"] = pd.to_datetime(df[["year", "month", "day"]])
            col = [c for c in df.columns if c not in
                   ("year", "month", "day", "date")][0]
            d = df.set_index("date")[col]
            out[name] = (1 + d).resample("ME").prod() - 1
        except Exception:
            pass
    return pd.DataFrame(out)


def to_monthly_returns(level):
    return level.resample("ME").last().pct_change()


def read_upload(file):
    """User CSV override: columns date,value (index level or return)."""
    df = pd.read_csv(file)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], dayfirst=True)
    s = df.set_index(df.columns[0]).iloc[:, 0].astype(float).sort_index()
    return s


# ----------------------------------------------------------------------------
# Hedge-fund sleeve construction (point-in-time)
# ----------------------------------------------------------------------------
def build_sleeve(panel, reg, cnpj_list, weighting, max_abs_ret):
    """Blend of the selected funds (equal or prior-month-AUM weighted).

    Returns the blended sleeve, a wide table of each fund's monthly returns,
    and a per-fund info table.
    """
    df = panel.merge(reg, on="CNPJ_FUNDO", how="left")
    df = df[df["CNPJ_FUNDO"].isin(cnpj_list)].sort_values(["CNPJ_FUNDO", "M"])
    df["ret"] = df.groupby("CNPJ_FUNDO")["VL_QUOTA"].pct_change()
    df["pl_prev"] = df.groupby("CNPJ_FUNDO")["VL_PATRIM_LIQ"].shift()
    # Only accept consecutive months; drop data-error quota jumps
    gap = df.groupby("CNPJ_FUNDO")["M"].diff().apply(
        lambda x: x.n if pd.notna(x) else np.nan)
    df.loc[gap != 1, "ret"] = np.nan
    df.loc[df["ret"].abs() > max_abs_ret, "ret"] = np.nan

    rows = []
    for m, g in df.dropna(subset=["ret"]).groupby("M"):
        if weighting == "Equal":
            r_ = g["ret"].mean()
        else:
            w = g["pl_prev"].fillna(0).values
            r_ = np.average(g["ret"], weights=w) if w.sum() > 0 else g["ret"].mean()
        rows.append({"M": m, "HF": r_, "n_funds": len(g)})
    sleeve = pd.DataFrame(rows).set_index("M").sort_index()
    sleeve.index = sleeve.index.to_timestamp(how="end").normalize()

    # Readable, unique labels per fund
    names = df.groupby("CNPJ_FUNDO")["DENOM_SOCIAL"].first()
    labels = {}
    for c, n in names.items():
        lab = short_name(n) if pd.notna(n) else \
            f"{c[:2]}.{c[2:5]}.{c[5:8]}/{c[8:12]}-{c[12:]}"
        labels[c] = lab if lab not in labels.values() else f"{lab} ({c[:8]})"

    wide = df.pivot(index="M", columns="CNPJ_FUNDO", values="ret")
    wide.index = wide.index.to_timestamp(how="end").normalize()
    wide = wide.rename(columns=labels)

    ok = df.dropna(subset=["ret"])
    info = ok.groupby("CNPJ_FUNDO").agg(
        fund=("DENOM_SOCIAL", "first"), manager=("GESTOR", "first"),
        first_month=("M", "min"), last_month=("M", "max"),
        last_aum_mm=("VL_PATRIM_LIQ", "last"))
    info["last_aum_mm"] /= 1e6
    info["first_month"] = info["first_month"].astype(str)
    info["last_month"] = info["last_month"].astype(str)
    return sleeve, wide, info.reset_index().rename(columns={"CNPJ_FUNDO": "cnpj"})


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def _nw_cov(y, lags):
    """Newey-West HAC covariance of the mean of y (T x k, demeaned)."""
    T = y.shape[0]
    S = y.T @ y / T
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = y[l:].T @ y[:-l] / T
        S += w * (G + G.T)
    return S


def sharpe_diff_test(e1, e2, lags=None):
    """Ledoit-Wolf (2008) HAC test of SR(e1) - SR(e2) on excess returns.
    Returns (diff_monthly, se, p_value). Annualize diff x sqrt(12)."""
    x = np.column_stack([e1, e2])
    x = x[~np.isnan(x).any(axis=1)]
    T = len(x)
    lags = lags if lags is not None else int(np.floor(4 * (T / 100) ** (2 / 9)))
    m = x.mean(0)
    g = (x ** 2).mean(0)
    sd = np.sqrt(g - m ** 2)
    diff = m[0] / sd[0] - m[1] / sd[1]
    grad = np.array([g[0] / sd[0] ** 3, -g[1] / sd[1] ** 3,
                     -0.5 * m[0] / sd[0] ** 3, 0.5 * m[1] / sd[1] ** 3])
    y = np.column_stack([x[:, 0] - m[0], x[:, 1] - m[1],
                         x[:, 0] ** 2 - g[0], x[:, 1] ** 2 - g[1]])
    Psi = _nw_cov(y, lags)
    se = np.sqrt(grad @ Psi @ grad / T)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(diff / se)))
    return diff, se, p


def ann_stats(r, rf):
    e = (r - rf).dropna()
    rr = r.loc[e.index]
    wealth = (1 + rr).cumprod()
    dd = wealth / wealth.cummax() - 1
    downside = e[e < 0].std() * np.sqrt(12)
    return {
        "Ann. return": (1 + rr).prod() ** (12 / len(rr)) - 1,
        "Ann. vol": rr.std() * np.sqrt(12),
        "Sharpe": e.mean() / e.std() * np.sqrt(12),
        "Sortino": e.mean() * 12 / downside if downside > 0 else np.nan,
        "Max DD": dd.min(),
        "Months": len(rr),
    }


def hac_ols(y, X, lags=6):
    X = sm.add_constant(X)
    d = pd.concat([y, X], axis=1).dropna()
    return sm.OLS(d.iloc[:, 0], d.iloc[:, 1:]).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})


def tangency_sharpe(ex):
    ex = ex.dropna()
    mu = ex.mean().values
    S = np.cov(ex.values, rowvar=False)
    return float(np.sqrt(mu @ np.linalg.solve(S, mu)) * np.sqrt(12))


def portfolio(rets, w):
    """Monthly-rebalanced portfolio return."""
    w = pd.Series(w)
    w = w[w != 0]
    d = rets[w.index].dropna()
    return d @ (w / w.sum())


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
NUM_FMT = ",.2f"  # #,##0.00


def line_chart(data, y_title="", x_title=None, height=380, y_fmt=NUM_FMT):
    """Altair line chart: #,##0.00 axis labels and a bold end-of-line label
    in each series' colour."""
    df = data.to_frame() if isinstance(data, pd.Series) else data.copy()
    temporal = isinstance(df.index, pd.DatetimeIndex)
    xname = "Date" if temporal else (x_title or "x")
    d = (df.rename_axis(xname).reset_index()
           .melt(xname, var_name="Series", value_name="Value").dropna())
    if d.empty:
        st.caption("No data to plot.")
        return
    x = alt.X(f"{xname}:{'T' if temporal else 'Q'}",
              title=None if temporal else x_title,
              axis=alt.Axis(format="%Y", tickCount="year") if temporal
              else alt.Axis(format=NUM_FMT))
    y = alt.Y("Value:Q", title=y_title, axis=alt.Axis(format=y_fmt),
              scale=alt.Scale(zero=False))
    color = alt.Color("Series:N", legend=alt.Legend(orient="bottom", title=None))
    lines = alt.Chart(d).mark_line(strokeWidth=2).encode(
        x=x, y=y, color=color,
        tooltip=["Series:N",
                 alt.Tooltip(f"{xname}:{'T' if temporal else 'Q'}",
                             format="%b %Y" if temporal else NUM_FMT),
                 alt.Tooltip("Value:Q", format=NUM_FMT)])
    ends = d.sort_values(xname).groupby("Series").tail(1)
    labels = alt.Chart(ends).mark_text(
        align="left", dx=6, fontWeight="bold", fontSize=12).encode(
        x=x, y=y, text=alt.Text("Value:Q", format=NUM_FMT),
        color=alt.Color("Series:N", legend=None))
    chart = (lines + labels).properties(
        height=height, padding={"left": 5, "top": 10, "right": 80, "bottom": 5})
    st.altair_chart(chart, width="stretch")


# ----------------------------------------------------------------------------
# Sizing, horizon and tail risk
# ----------------------------------------------------------------------------
def _sharpe_grid(eb, eh, grid):
    """Annualised Sharpe of (1-s)*baseline + s*fund for each s (excess rets)."""
    mb, mh = eb.mean(), eh.mean()
    vb, vh = eb.var(ddof=1), eh.var(ddof=1)
    c = np.cov(eb, eh)[0, 1]
    m = (1 - grid) * mb + grid * mh
    v = (1 - grid) ** 2 * vb + grid ** 2 * vh + 2 * grid * (1 - grid) * c
    return m / np.sqrt(v) * np.sqrt(12)


def optimal_sleeve(eb, eh, grid, n_boot=500, block=6, seed=0):
    """In-sample Sharpe-maximising sleeve plus a stationary block-bootstrap
    10-90% range for the optimum (shows how noisy it is)."""
    eb, eh = np.asarray(eb), np.asarray(eh)
    curve = _sharpe_grid(eb, eh, grid)
    T = len(eb)
    rng = np.random.default_rng(seed)
    opts = []
    for _ in range(n_boot):
        idx, i = [], rng.integers(T)
        while len(idx) < T:
            idx.append(i)
            i = rng.integers(T) if rng.random() < 1 / block else (i + 1) % T
        opts.append(grid[np.nanargmax(_sharpe_grid(eb[idx], eh[idx], grid))])
    return curve, grid[np.nanargmax(curve)], np.percentile(opts, [10, 50, 90])


def horizon_years(sr_ann, z):
    """Years for P(outperform) to reach the chosen confidence, assuming iid
    normal log returns: P = N(SR * sqrt(T))  =>  T = (z / SR)^2."""
    return (z / sr_ann) ** 2 if sr_ann > 0 else np.inf


def tail_stats(r):
    r = r.dropna()
    q = r.quantile(0.05)
    return {"Skewness": r.skew(), "Excess kurtosis": r.kurt(),
            "Worst month": r.min(), "Worst month date": r.idxmin().strftime("%b %Y"),
            "CVaR 5% (monthly)": r[r <= q].mean(), "% negative months": (r < 0).mean(),
            "Months": len(r)}


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def _pct(x):
    return f"{x:.1%}"


def _sig(p):
    return "significant at 5%" if p < 0.05 else (
        "significant at 10%" if p < 0.10 else "not statistically significant")


def main():
    st.set_page_config("BR hedge funds vs Sharpe", layout="wide")
    st.title("Does this fund improve my portfolio's Sharpe?")

    with st.sidebar:
        st.header("Period")
        st.caption("Pick any day; the analysis uses whole months.")
        d0 = st.date_input("Start (year-month)", date(2010, 1, 1),
                           min_value=date(2005, 1, 1), max_value=date.today(),
                           format="YYYY/MM/DD")
        d1 = st.date_input("End (year-month)", date.today(),
                           min_value=date(2005, 1, 1), max_value=date.today(),
                           format="YYYY/MM/DD")
        start_m = pd.Period(d0, "M")
        end_m = pd.Period(d1, "M")
        if pd.Timestamp(d1) < end_m.to_timestamp(how="end").normalize():
            end_m -= 1  # drop an incomplete final month
        st.caption(f"Window: **{start_m.strftime('%b %Y')} – "
                   f"{end_m.strftime('%b %Y')}**")

        st.header("Funds")
        raw_cnpjs = st.text_area(
            "Fund CNPJs (one per line)",
            placeholder="22.187.946/0001-41\n12.808.980/0001-32 ; My label",
            help="Names come from the CVM registry. To set your own label, "
                 "add it after the CNPJ (e.g. '12.808.980/0001-32 ; SPX').")
        cnpjs, manual_names = [], {}
        for line in raw_cnpjs.splitlines():
            mt = re.match(r"\s*([\d./-]{11,20})\s*[;,|]?\s*(.*)$", line)
            if not mt or not re.sub(r"\D", "", mt.group(1)):
                continue
            c = re.sub(r"\D", "", mt.group(1)).zfill(14)
            cnpjs.append(c)
            if mt.group(2).strip():
                manual_names[c] = mt.group(2).strip()
        weighting = st.radio("Sleeve weighting (if several funds)",
                             ["Equal", "AUM"])

        st.header("Baseline portfolio (%)")
        w_cdi = st.number_input("CDI", 0, 100, 40, step=5)
        w_imab = st.number_input("IMA-B", 0, 100, 20, step=5)
        w_ibov = st.number_input("Ibovespa", 0, 100, 20, step=5)
        w_world = st.number_input("World equity (BRL)", 0, 100, 20, step=5)
        w_sum = w_cdi + w_imab + w_ibov + w_world
        st.caption(f"Total: **{w_sum}%**" + ("" if w_sum == 100 else " ⚠️ must be 100%"))
        sleeves = st.multiselect("HF sleeve sizes (%)",
                                 [5, 10, 15, 20, 25, 30, 40, 50], [10, 20, 30])
        conf = st.slider("Horizon confidence (%)", 50, 99, 90,
                         help="Used in the Horizon tab: years needed for the "
                              "fund to beat CDI with this probability.")

        run = st.button("Run analysis", type="primary", width="stretch")

        st.divider()
        with st.expander("Data sources & advanced", expanded=False):
            cdi_code = st.text_input("SGS: CDI monthly %", "4391")
            imab_code = st.text_input("SGS: IMA-B index", "12466")
            ptax_code = st.text_input("SGS: PTAX USD/BRL", "1")
            ibov_tk = st.text_input("Yahoo: Ibovespa", "^BVSP")
            world_tk = st.text_input("Yahoo: World ETF (USD)", "URTH")
            st.caption("Optional CSV overrides (date,value). Levels or "
                       "returns: values > 2 are treated as index levels.")
            up = {k: st.file_uploader(k, type="csv", key=k)
                  for k in ["IMA-B", "Ibovespa", "World (BRL)"]}
            max_abs = st.number_input(
                "Data-error filter: ignore fund monthly returns beyond ± (%)",
                10, 500, 50, step=10,
                help="CVM quota series sometimes contain errors (wrong decimal, "
                     "quota splits/consolidations). A month where a fund's "
                     "quota moves more than this is treated as bad data and "
                     "excluded for that fund.") / 100

    if not run:
        st.info("Set parameters and click **Run analysis**. The first run "
                "downloads several GB of CVM files and can take a while; "
                "later runs use ./data_cache.")
        return
    if not cnpjs:
        st.error("Enter at least one fund CNPJ.")
        return
    if (end_m - start_m).n < 24:
        st.error("Use a window of at least 24 months.")
        return
    if w_sum != 100:
        st.error(f"Baseline weights add up to {w_sum}%; they must total 100%.")
        return
    if not sleeves:
        st.error("Pick at least one HF sleeve size.")
        return

    start = (start_m - 1).to_timestamp().strftime("%Y-%m-%d")
    end = end_m.to_timestamp(how="end").strftime("%Y-%m-%d")

    # --- Registry & universe ------------------------------------------------
    with st.spinner("Loading CVM fund registry…"):
        reg = load_registry()
    reg = enrich_registry(reg, cnpjs, manual_names)
    unknown = [c for c in cnpjs if c not in set(reg["CNPJ_FUNDO"])]
    if unknown:
        st.warning("Not in the CVM registries (shown by CNPJ unless you add "
                   f"a label next to it): {', '.join(unknown)}")
    universe = frozenset(cnpjs)

    # Only fetch years that can contain these funds' data
    first, last_c = fund_date_bounds(reg, cnpjs)
    y0 = (start_m - 1).year
    if first is not None and first.year > y0:
        y0 = first.year
    t1 = end_m.to_timestamp(how="end")
    if last_c is not None and last_c < t1:
        t1 = last_c
    prog = st.progress(0.0)
    panel = load_fund_panel(y0, t1, universe, prog)
    prog.empty()
    if panel.empty:
        st.error("No CVM data found for this universe.")
        return

    sleeve, fund_rets, info = build_sleeve(panel, reg, cnpjs, weighting,
                                           max_abs)

    # --- Benchmarks ---------------------------------------------------------
    with st.spinner("Loading BCB / market data…"):
        try:
            cdi = sgs(cdi_code, start, end) / 100
        except Exception as e:
            st.error(f"Couldn't download CDI from the Central Bank: {e}. "
                     "BCB's API is sometimes down; wait a minute and run again.")
            return
        cdi.index = cdi.index.to_period("M").to_timestamp(how="end").normalize()

        def level_or_upload(key, fetch):
            if up[key] is not None:
                s = read_upload(up[key])
                return to_monthly_returns(s) if s.abs().median() > 2 else \
                    s.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            return fetch()

        try:
            imab = level_or_upload(
                "IMA-B", lambda: to_monthly_returns(sgs(imab_code, start, end)))
        except Exception as e:
            st.error(f"Couldn't download IMA-B from the Central Bank: {e}. Run "
                     "again in a minute, or upload ANBIMA's IMA-B CSV under "
                     "'Data sources & advanced'.")
            return
        ibov = level_or_upload(
            "Ibovespa", lambda: yahoo_monthly_close(ibov_tk, start, end).pct_change())
        try:
            ptax = sgs(ptax_code, start, end).resample("ME").last()
        except Exception as e:
            st.warning(f"PTAX unavailable from BCB ({e}); using USD/BRL from "
                       "Yahoo Finance instead.")
            ptax = yahoo_monthly_close("BRL=X", start, end)

        def world_brl():
            usd = yahoo_monthly_close(world_tk, start, end)
            return (usd * ptax.reindex(usd.index)).pct_change()

        world = level_or_upload("World (BRL)", world_brl)
        fx = ptax.pct_change()

    R = pd.concat({"CDI": cdi, "IMA-B": imab, "Ibovespa": ibov,
                   "World": world, "HF": sleeve["HF"]}, axis=1)
    R = R.loc[start_m.to_timestamp():end_m.to_timestamp(how="end")].dropna()
    if len(R) < 24:
        st.error(f"Only {len(R)} months with data for every series.")
        return
    rf = R["CDI"]
    st.caption(f"Sample: {R.index[0]:%b %Y} – {R.index[-1]:%b %Y} "
               f"({len(R)} months)")

    base_w = {"CDI": w_cdi / 100, "IMA-B": w_imab / 100,
              "Ibovespa": w_ibov / 100, "World": w_world / 100}
    ports = {"Baseline": portfolio(R, base_w)}
    for pct in sorted(sleeves):
        s = pct / 100
        w = {k: v * (1 - s) for k, v in base_w.items()}
        w["HF"] = s
        ports[f"Baseline + {pct}% HF"] = portfolio(R, w)
    P = pd.DataFrame(ports)
    last = P.columns[-1]
    ex = R.sub(rf, axis=0).drop(columns="CDI")
    risky = ["IMA-B", "Ibovespa", "World"]

    tabs = st.tabs(["Performance", "Summary", "Sharpe tests", "Optimal size",
                    "Horizon", "Correlations", "Tail risk",
                    "Spanning & factors", "Robustness", "Funds"])
    # Each fund aligned with the benchmarks over the window (own history)
    win = slice(start_m.to_timestamp(), end_m.to_timestamp(how="end"))
    bench = pd.concat({"CDI": cdi, "IMA-B": imab, "Ibovespa": ibov,
                       "World": world}, axis=1).loc[win].dropna()
    fr_all = fund_rets.reindex(bench.index)
    base_full = portfolio(bench, base_w)
    series = {c: fr_all[c] for c in fr_all.columns if fr_all[c].count() >= 24}
    if fr_all.shape[1] > 1:
        series["HF blend"] = R["HF"]
    fmt = {"Ann. return": "{:.2%}", "Ann. vol": "{:.2%}", "Sharpe": "{:.2f}",
           "Sortino": "{:.2f}", "Max DD": "{:.1%}", "Months": "{:.0f}"}

    # --- Performance ----------------------------------------------------------
    with tabs[0]:
        fr = fund_rets.loc[start_m.to_timestamp():end_m.to_timestamp(how="end")]
        firsts = fr.apply(lambda c: c.first_valid_index())
        firsts = firsts.dropna()
        if firsts.empty:
            st.warning("No fund returns in the selected window.")
        else:
            # Start where every fund has data (or the chosen start, if later)
            t0 = firsts.max()
            fr = fr.loc[t0:].dropna(axis=1, how="all")
            ref = pd.concat({"CDI": cdi, "Ibovespa": ibov}, axis=1).reindex(fr.index)
            lv = (1 + pd.concat([fr, ref], axis=1).fillna(0)).cumprod() * 100
            base_row = pd.DataFrame(100.0, index=[t0 - pd.offsets.MonthEnd(1)],
                                    columns=lv.columns)
            lv = pd.concat([base_row, lv])
            # Stop a fund's line (and CAGR) at its last reported month
            lasts = fr.apply(lambda c: c.last_valid_index())
            for c, t in lasts.items():
                lv.loc[lv.index > t, c] = np.nan
            n_m = pd.concat([fr.notna().sum(), pd.Series(len(fr), index=ref.columns)])
            cagr = (lv.ffill().iloc[-1] / 100) ** (12 / n_m) - 1
            fund_cagr = cagr[fr.columns].sort_values()
            st.info(
                f"Since **{base_row.index[0]:%b %Y}** ({len(fr)} months, the "
                f"first month all selected funds have data), the best fund "
                f"compounded **{_pct(fund_cagr.iloc[-1])}/yr** "
                f"(*{fund_cagr.index[-1]}*) and the worst {_pct(fund_cagr.iloc[0])}/yr "
                f"(*{fund_cagr.index[0]}*), vs CDI {_pct(cagr['CDI'])}/yr and "
                f"Ibovespa {_pct(cagr['Ibovespa'])}/yr.")
            st.subheader("Cumulative return (rebased to 100)")
            line_chart(lv, "Index (start = 100)", y_fmt=",.0f")

    # --- Summary ------------------------------------------------------------
    with tabs[1]:
        stats = pd.DataFrame({k: ann_stats(P[k], rf) for k in P}).T
        assets = pd.DataFrame({k: ann_stats(R[k], rf)
                               for k in ["CDI", "IMA-B", "Ibovespa", "World", "HF"]}).T
        best = stats["Sharpe"].idxmax()
        st.info(
            f"Baseline Sharpe is **{stats.loc['Baseline','Sharpe']:.2f}**; with "
            f"the largest sleeve ({last}) it is **{stats.loc[last,'Sharpe']:.2f}**, "
            f"and the best of all mixes is *{best}*. "
            f"The HF sleeve alone returned {_pct(assets.loc['HF','Ann. return'])}/yr "
            f"vs CDI {_pct(assets.loc['CDI','Ann. return'])}, with max drawdown "
            f"{_pct(assets.loc['HF','Max DD'])} (baseline: "
            f"{_pct(stats.loc['Baseline','Max DD'])}).")
        st.subheader("Portfolios")
        st.dataframe(stats.style.format(fmt))
        st.subheader("Building blocks")
        st.dataframe(assets.style.format(fmt))
        st.subheader("Cumulative excess return over CDI (start = 100)")
        line_chart((1 + P.sub(rf, axis=0)).cumprod() * 100, "Index", y_fmt=",.0f")

    # --- Sharpe tests -------------------------------------------------------
    with tabs[2]:
        rows = []
        base_ex = (P["Baseline"] - rf).values
        for k in P.columns[1:]:
            d, se, p = sharpe_diff_test((P[k] - rf).values, base_ex)
            rows.append({"Portfolio": k, "ΔSharpe (ann.)": d * np.sqrt(12),
                         "SE (ann.)": se * np.sqrt(12), "p-value": p})
        T = pd.DataFrame(rows).set_index("Portfolio")
        kb = T["ΔSharpe (ann.)"].abs().idxmax()
        st.info(
            f"The biggest change is **{T.loc[kb,'ΔSharpe (ann.)']:+.2f}** "
            f"Sharpe points for *{kb}* (p = {T.loc[kb,'p-value']:.3f}), which is "
            f"**{_sig(T.loc[kb,'p-value'])}**. "
            + ("Adding hedge funds measurably changed risk-adjusted return."
               if T["p-value"].min() < 0.05 else
               "The data can't distinguish the HF portfolios from the baseline."))
        st.markdown("Ledoit-Wolf (2008) HAC test, H0: Sharpe equal to baseline.")
        st.dataframe(T.style.format({"ΔSharpe (ann.)": "{:+.3f}",
                                     "SE (ann.)": "{:.3f}", "p-value": "{:.3f}"}))

    # --- Optimal size -------------------------------------------------------
    with tabs[3]:
        grid = np.linspace(0, 1, 101)
        curves, rows = {}, []
        for name, fr in series.items():
            d = pd.concat([fr, base_full, bench["CDI"]], axis=1).dropna()
            eh, eb = d.iloc[:, 0] - d.iloc[:, 2], d.iloc[:, 1] - d.iloc[:, 2]
            curve, s_opt, (p10, p50, p90) = optimal_sleeve(eb.values, eh.values, grid)
            curves[name] = pd.Series(curve, index=grid * 100)
            rows.append({"Fund": name, "Optimal sleeve": s_opt,
                         "Bootstrap 10%": p10, "Bootstrap median": p50,
                         "Bootstrap 90%": p90, "Sharpe baseline": curve[0],
                         "Sharpe at optimum": curve.max(),
                         "Months": len(d)})
        O = pd.DataFrame(rows).set_index("Fund")
        if O.empty:
            st.warning("No fund has 24+ months of data in the window.")
        else:
            top = O["Sharpe at optimum"].sub(O["Sharpe baseline"]).idxmax()
            st.info(
                f"Maximising Sharpe, the best allocation for *{top}* is "
                f"**{O.loc[top,'Optimal sleeve']:.0%}** of the portfolio (Sharpe "
                f"{O.loc[top,'Sharpe baseline']:.2f} → {O.loc[top,'Sharpe at optimum']:.2f}). "
                f"The bootstrap range ({O.loc[top,'Bootstrap 10%']:.0%}–"
                f"{O.loc[top,'Bootstrap 90%']:.0%}) shows how uncertain that optimum "
                f"is; treat a wide range as a weak signal and size toward the low end.")
            st.dataframe(O.style.format({
                "Optimal sleeve": "{:.0%}", "Bootstrap 10%": "{:.0%}",
                "Bootstrap median": "{:.0%}", "Bootstrap 90%": "{:.0%}",
                "Sharpe baseline": "{:.2f}", "Sharpe at optimum": "{:.2f}",
                "Months": "{:.0f}"}))
            st.subheader("Portfolio Sharpe by sleeve size")
            line_chart(pd.DataFrame(curves), "Sharpe (ann.)", "Sleeve (%)")
            st.caption("The fund replaces the baseline pro rata (monthly "
                       "rebalancing), over each fund's own history in the window. "
                       "Bootstrap: 500 stationary block resamples (6-month blocks).")

    # --- Horizon -------------------------------------------------------------
    with tabs[4]:
        z = norm.ppf(conf / 100)
        rows, probs = [], {}
        yrs = np.round(np.arange(0.5, 10.01, 0.25), 2)
        for name, fr in series.items():
            d = pd.concat([fr, base_full, bench["CDI"]], axis=1).dropna()
            lf, lb, lc = np.log1p(d.iloc[:, 0]), np.log1p(d.iloc[:, 1]), np.log1p(d.iloc[:, 2])
            x_cdi, x_base = lf - lc, lf - lb
            sr_c = x_cdi.mean() / x_cdi.std() * np.sqrt(12)
            ir_b = x_base.mean() / x_base.std() * np.sqrt(12)
            rows.append({"Fund": name, "Sharpe vs CDI (log)": sr_c,
                         "Ann. vol": np.log1p(d.iloc[:, 0]).std() * np.sqrt(12),
                         f"Years to beat CDI ({conf}%)": horizon_years(sr_c, z),
                         "Info ratio vs baseline": ir_b,
                         f"Years to beat baseline ({conf}%)": horizon_years(ir_b, z)})
            probs[name] = pd.Series(norm.cdf(sr_c * np.sqrt(yrs)) * 100, index=yrs)
        H = pd.DataFrame(rows).set_index("Fund")
        if not H.empty:
            hc = H[f"Years to beat CDI ({conf}%)"]
            fin = hc[np.isfinite(hc)]
            st.info(
                (f"To beat CDI with {conf}% confidence, the shortest horizon is "
                 f"**{fin.min():.1f} years** (*{fin.idxmin()}*)"
                 + (f" and the longest {fin.max():.1f} years (*{fin.idxmax()}*)."
                    if len(fin) > 1 else ".")
                 if len(fin) else
                 f"No fund beat CDI on a risk-adjusted basis in this window, so "
                 f"no horizon reaches {conf}% confidence.")
                + " Funds needing longer than you'd realistically hold them are "
                  "a bet you can't expect to see pay off.")
            fmt_h = lambda v: "never" if not np.isfinite(v) else f"{v:,.1f}"
            st.dataframe(H.style.format({
                "Sharpe vs CDI (log)": "{:.2f}", "Ann. vol": "{:.1%}",
                "Info ratio vs baseline": "{:.2f}",
                f"Years to beat CDI ({conf}%)": fmt_h,
                f"Years to beat baseline ({conf}%)": fmt_h}))
            st.markdown(
                r"Horizon $T = \left(\dfrac{z_c}{SR}\right)^2$, from "
                r"$P(\text{fund} > \text{CDI over } T) = \Phi(SR\sqrt{T})$ with "
                r"$SR = \mu/\sigma$ of annual excess log returns. Higher vol at the "
                r"same excess return ⇒ lower SR ⇒ longer horizon.")
            st.caption("Assumes iid normal returns and treats the historical "
                       "Sharpe as the true one, so it understates the horizon "
                       "when the Sharpe is estimated from a short history.")
            st.subheader("Probability of beating CDI by holding period (%)")
            line_chart(pd.DataFrame(probs), "%", "Years held", y_fmt=",.0f")

    # --- Correlations ---------------------------------------------------------
    with tabs[5]:
        C = pd.concat([fr_all[list(c for c in fr_all.columns)],
                       bench[["IMA-B", "Ibovespa", "World"]],
                       fx.reindex(bench.index).rename("USD/BRL"),
                       base_full.rename("Baseline")], axis=1)
        C = C.dropna(axis=1, thresh=24)
        corr = C.corr(min_periods=24)
        fcols = [c for c in fr_all.columns if c in corr.columns]
        if fcols:
            sub_c = corr.loc[fcols, ["IMA-B", "Ibovespa", "World", "USD/BRL", "Baseline"]]
            pair = None
            if len(fcols) > 1:
                ff = corr.loc[fcols, fcols].where(
                    np.triu(np.ones((len(fcols),) * 2, bool), 1)).stack()
                pair = ff.idxmax(), ff.max()
            st.info(
                f"On average the selected fund(s) correlate **{sub_c['Baseline'].mean():.2f}** "
                f"with your baseline, {sub_c['Ibovespa'].mean():.2f} with Ibovespa and "
                f"{sub_c['USD/BRL'].mean():.2f} with USD/BRL. "
                + (f"The most similar pair is *{pair[0][0]}* / *{pair[0][1]}* "
                   f"({pair[1]:.2f}); above ~0.7 they add little diversification "
                   "to each other." if pair else
                   "Lower correlation with the baseline means more diversification "
                   "per unit of risk."))
        st.dataframe(corr.style.format("{:.2f}").background_gradient(
            cmap="RdBu_r", vmin=-1, vmax=1), width="stretch")
        st.caption("Monthly returns, pairwise over overlapping months (min. 24). "
                   "CDI is left out: it barely moves, so its correlations are "
                   "not meaningful.")

    # --- Tail risk ------------------------------------------------------------
    with tabs[6]:
        tseries = dict(series)
        tseries.update({"IMA-B": bench["IMA-B"], "Ibovespa": bench["Ibovespa"],
                        "World": bench["World"], "Baseline": base_full,
                        last: P[last]})
        TR = pd.DataFrame({k: tail_stats(v) for k, v in tseries.items()}).T
        fnames = [k for k in series]
        if fnames:
            TRf = TR.loc[fnames]
            wk = TRf["Worst month"].astype(float).idxmin()
            neg = [k for k in fnames if TR.loc[k, "Skewness"] < -0.5]
            st.info(
                f"The worst single month among the funds was **{TR.loc[wk,'Worst month']:.1%}** "
                f"(*{wk}*, {TR.loc[wk,'Worst month date']}), vs Ibovespa's "
                f"{TR.loc['Ibovespa','Worst month']:.1%}. "
                + (f"{', '.join(neg)} show clearly negative skew (< −0.5): "
                   "smooth gains with occasional sharp losses, which Sharpe understates."
                   if neg else "No fund shows strongly negative skew, so Sharpe "
                   "isn't hiding obvious crash risk."))
        st.dataframe(TR.style.format({
            "Skewness": "{:.2f}", "Excess kurtosis": "{:.2f}",
            "Worst month": "{:.2%}", "CVaR 5% (monthly)": "{:.2%}",
            "% negative months": "{:.0%}", "Months": "{:.0f}"}), width="stretch")
        st.caption("Skew < 0 and high kurtosis = fat left tail. CVaR 5% = "
                   "average return in the worst 5% of months.")

    # --- Spanning / factors ---------------------------------------------------
    with tabs[7]:
        m1 = hac_ols(ex["HF"], ex[risky])
        a, se_eps = m1.params["const"], np.sqrt(m1.scale)
        sr_base = tangency_sharpe(ex[risky])
        ar = a / se_eps * np.sqrt(12)
        sr_new = np.sqrt(sr_base**2 + ar**2) if a > 0 else sr_base

        F = ex[risky].copy()
        F["USD/BRL"] = fx.reindex(F.index)
        nf = nefin_factors()
        if not nf.empty:
            F = F.join(nf.drop(columns="Market", errors="ignore"), how="left")
        F = F.dropna(axis=1, how="all")
        m2 = hac_ols(ex["HF"], F)
        a2, t2 = m2.params["const"] * 12, m2.tvalues["const"]

        st.info(
            f"Against the baseline assets, HF alpha is **{_pct(a*12)}/yr** "
            f"(t = {m1.tvalues['const']:.2f}); in-sample it would lift the best "
            f"achievable Sharpe from {sr_base:.2f} to {sr_new:.2f}. "
            f"Adding FX{' and NEFIN' if not nf.empty else ''} factors, alpha is "
            f"{_pct(a2)}/yr (t = {t2:.2f}), so it "
            + ("**survives** the factor controls." if abs(t2) >= 2 and a2 > 0 else
               "is **not robust** to factor controls (much of it looks like "
               "replicable exposure or noise)."))
        st.subheader("Spanning regression: HF excess on baseline assets")
        st.text(m1.summary().tables[1].as_text())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Alpha (ann.)", _pct(a * 12))
        c2.metric("Appraisal ratio (ann.)", f"{ar:.2f}")
        c3.metric("Tangency Sharpe, base", f"{sr_base:.2f}")
        c4.metric("Tangency Sharpe, + HF", f"{sr_new:.2f}")
        st.caption("Positive, significant alpha ⇒ HF expands the efficient "
                   "frontier: SR²(new) = SR²(old) + (α/σε)². In-sample "
                   "tangency Sharpes are upper bounds.")
        st.subheader("Factor regression (is the alpha cheap beta?)")
        if nf.empty:
            st.warning("NEFIN factors unavailable; using baseline assets + FX.")
        st.text(m2.summary().tables[1].as_text())
        st.caption(f"R² = {m2.rsquared:.2f}, N = {int(m2.nobs)}")

    # --- Robustness -----------------------------------------------------------
    with tabs[8]:
        cuts = [("2010-01", "2014-12"), ("2015-01", "2019-12"),
                ("2020-01", "2022-12"), ("2023-01", "2099-12")]
        sub = []
        for s0, s1 in cuts:
            Ps, rfs = P.loc[s0:s1], rf.loc[s0:s1]
            if len(Ps) < 24:
                continue
            row = {"Period": f"{Ps.index[0]:%Y}–{Ps.index[-1]:%Y}"}
            for k in P:
                e = Ps[k] - rfs
                row[k] = e.mean() / e.std() * np.sqrt(12)
            _, _, p = sharpe_diff_test((Ps[last] - rfs).values,
                                       (Ps["Baseline"] - rfs).values)
            row[f"p ({last} vs base)"] = p
            mm_ = hac_ols(ex["HF"].loc[s0:s1], ex[risky].loc[s0:s1])
            row["HF alpha ann."] = mm_.params["const"] * 12
            row["alpha t"] = mm_.tvalues["const"]
            sub.append(row)
        S = pd.DataFrame(sub).set_index("Period") if sub else pd.DataFrame()

        alphas = {}
        for i in range(36, len(ex) + 1):
            w_ = ex.iloc[i - 36:i]
            alphas[w_.index[-1]] = sm.OLS(
                w_["HF"], sm.add_constant(w_[risky])).fit().params["const"] * 12
        alphas = pd.Series(alphas, name="alpha")
        worst = R["Ibovespa"] <= R["Ibovespa"].quantile(0.1)

        n_pos = int(((S["HF alpha ann."] > 0) & (S["alpha t"] >= 2)).sum()) if len(S) else 0
        st.info(
            f"HF alpha was positive and significant (t ≥ 2) in **{n_pos} of "
            f"{len(S)}** sub-periods"
            + (f"; the latest 36-month alpha is {_pct(alphas.iloc[-1])}/yr."
               if len(alphas) else ".")
            + f" In Ibovespa's worst 10% of months the sleeve averaged "
            f"{_pct(R.loc[worst,'HF'].mean())} vs Ibovespa "
            f"{_pct(R.loc[worst,'Ibovespa'].mean())}, a measure of crisis "
            f"diversification.")
        if len(S):
            st.subheader("Sub-periods")
            st.dataframe(S.style.format("{:.2f}"))
        st.subheader("Rolling 36-month Sharpe")
        roll = P.sub(rf, axis=0).rolling(36)
        line_chart(roll.mean() / roll.std() * np.sqrt(12), "Sharpe")
        st.subheader("Rolling 36-month HF alpha (ann., %)")
        line_chart(alphas.rename("HF alpha") * 100, "% per year", y_fmt=",.0f")
        st.subheader("Drawdowns (%)")
        wealth = (1 + P).cumprod()
        line_chart((wealth / wealth.cummax() - 1) * 100, "%", y_fmt=",.0f")
        st.subheader("Returns in Ibovespa's worst months")
        st.write(pd.DataFrame({
            "Avg return, worst 10% Ibov months": R[worst][["Ibovespa", "HF"]].mean(),
            "Avg return, other months": R[~worst][["Ibovespa", "HF"]].mean(),
        }).style.format("{:.2%}"))

    # --- Funds --------------------------------------------------------------
    with tabs[9]:
        dead = info[info["last_month"] < str(end_m)]
        st.info(
            f"**{len(info)}** fund(s) with data. "
            + (f"{len(dead)} stopped reporting before {end_m.strftime('%b %Y')} "
               "(closed, merged or migrated) and are kept until their last month."
               if len(dead) else "All report through the end of the sample.")
            + f" The blended sleeve holds {sleeve['n_funds'].loc[R.index].mean():.1f} "
              "funds on average.")
        st.dataframe(info.style.format({"last_aum_mm": "R$ {:,.0f} mm"}),
                     width="stretch")
        st.subheader("Number of live funds in sleeve")
        line_chart(sleeve["n_funds"].loc[win].rename("Live funds"), "Funds")
        st.download_button("Download monthly returns (CSV)",
                           pd.concat([R, P, fund_rets], axis=1).to_csv(),
                           "monthly_returns.csv")

def build_cache(start_year=2009):
    """Pre-build the month-end cache (run locally, then commit data_cache/
    me_*.parquet to the repo so Streamlit Cloud never downloads CVM files)."""
    t0 = time.time()
    panel_files = _daily_file_list(start_year, pd.Timestamp(date.today()))
    print(f"Building {len(panel_files)} files into {CACHE} …")
    load_registry()
    load_fund_panel(start_year, pd.Timestamp(date.today()), [], None)
    print(f"Done in {time.time() - t0:,.0f}s")


if __name__ == "__main__":
    if "--build-cache" in sys.argv:
        i = sys.argv.index("--build-cache")
        build_cache(int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 2009)
    else:
        main()
