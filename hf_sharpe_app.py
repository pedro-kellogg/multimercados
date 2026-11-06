"""
Do Brazilian hedge funds (multimercados) from large managers improve a
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

import hashlib
import io
import os
import re
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st
import statsmodels.api as sm

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
os.makedirs(CACHE, exist_ok=True)

CVM_DAILY = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/"
CVM_CAD = "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv"
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
            "INVEST_QUALIF"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["CNPJ_FUNDO"] = df["CNPJ_FUNDO"].str.replace(r"\D", "", regex=True)
    df = df.dropna(subset=["CNPJ_FUNDO"]).drop_duplicates("CNPJ_FUNDO", keep="last")
    return df


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


def _parse_daily_zip(content, cnpjs):
    """Month-end quota, AUM and quotaholders for the given CNPJs."""
    out = []
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with z.open(name) as f:
                df = pd.read_csv(
                    f, sep=";", encoding="latin-1", dtype=str, low_memory=False,
                    usecols=lambda c: c.startswith("CNPJ_FUNDO") or c in
                    {"DT_COMPTC", "VL_QUOTA", "VL_PATRIM_LIQ", "NR_COTST"})
            # Pre/post CVM 175 column names (CNPJ_FUNDO vs CNPJ_FUNDO_CLASSE)
            ren = {}
            for c in df.columns:
                if c.startswith("CNPJ_FUNDO"):
                    ren[c] = "CNPJ_FUNDO"
            df = df.rename(columns=ren)
            df["CNPJ_FUNDO"] = df["CNPJ_FUNDO"].str.replace(r"\D", "", regex=True)
            df = df[df["CNPJ_FUNDO"].isin(list(cnpjs))]
            if df.empty:
                continue
            df["DT"] = pd.to_datetime(df["DT_COMPTC"], errors="coerce")
            for c in ["VL_QUOTA", "VL_PATRIM_LIQ", "NR_COTST"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            # Post-175 files can carry subclasses: keep the largest row
            df = (df.sort_values("VL_PATRIM_LIQ")
                    .drop_duplicates(["CNPJ_FUNDO", "DT"], keep="last"))
            df["M"] = df["DT"].dt.to_period("M")
            df = (df.sort_values("DT")
                    .groupby(["CNPJ_FUNDO", "M"]).tail(1))
            out.append(df[["CNPJ_FUNDO", "M", "DT", "VL_QUOTA",
                           "VL_PATRIM_LIQ", "NR_COTST"]])
    return pd.concat(out) if out else pd.DataFrame()


def load_fund_panel(start_year, end_date, cnpjs, progress):
    files = _daily_file_list(start_year, end_date)
    frames = []
    key = hashlib.md5(",".join(sorted(cnpjs)).encode()).hexdigest()[:10]
    for i, (url, fname, label) in enumerate(files):
        progress.progress((i + 1) / len(files), text=f"CVM daily data: {label}")
        pq = os.path.join(CACHE, f"panel_{label}_{key}.parquet")
        if os.path.exists(pq):
            frames.append(pd.read_parquet(pq))
            continue
        try:
            content = _cached_bytes(url, fname)
        except Exception as e:  # file not yet published, etc.
            st.warning(f"Skipped {label}: {e}")
            continue
        part = _parse_daily_zip(content, cnpjs)
        if not part.empty:
            part = part.astype({"M": str})
            part.to_parquet(pq)
            frames.append(part)
    panel = pd.concat(frames, ignore_index=True)
    panel["M"] = pd.PeriodIndex(panel["M"], freq="M")
    panel = (panel.sort_values("DT")
                  .drop_duplicates(["CNPJ_FUNDO", "M"], keep="last"))
    return panel


# ----------------------------------------------------------------------------
# BCB / Yahoo / NEFIN
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def sgs(code, start, end):
    """SGS series, fetched in <=10y chunks (API limit for daily series)."""
    frames = []
    s = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while s <= end:
        e = min(s + pd.DateOffset(years=10) - pd.Timedelta(days=1), end)
        params = {"formato": "json",
                  "dataInicial": s.strftime("%d/%m/%Y"),
                  "dataFinal": e.strftime("%d/%m/%Y")}
        r = requests.get(BCB_SGS.format(code=code), params=params,
                         headers=HEADERS, timeout=120)
        r.raise_for_status()
        frames.append(pd.DataFrame(r.json()))
        s = e + pd.Timedelta(days=1)
    df = pd.concat(frames)
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df.set_index("data")["valor"].sort_index()


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
def build_sleeve(panel, reg, top_n, cnpj_list, weighting, max_abs_ret):
    """Point-in-time HF sleeve.

    Top-N mode: each January, rank managers by prior-December multimercado AUM
    and hold one flagship per manager. CNPJ mode: hold the listed funds
    whenever they report.
    """
    df = panel.merge(reg, on="CNPJ_FUNDO", how="left")
    df = df.sort_values(["CNPJ_FUNDO", "M"])
    df["ret"] = df.groupby("CNPJ_FUNDO")["VL_QUOTA"].pct_change()
    df["pl_prev"] = df.groupby("CNPJ_FUNDO")["VL_PATRIM_LIQ"].shift()
    # Only accept consecutive months; drop data-error quota jumps
    gap = df.groupby("CNPJ_FUNDO")["M"].diff().apply(
        lambda x: x.n if pd.notna(x) else np.nan)
    df.loc[gap != 1, "ret"] = np.nan
    df.loc[df["ret"].abs() > max_abs_ret, "ret"] = np.nan
    df["year"] = df["M"].dt.year
    years = sorted(df["year"].unique())
    rows, picks = [], []

    def add_month(m, g, w0=None):
        g = g.dropna(subset=["ret"])  # dead funds drop out when they stop
        if g.empty:
            return
        if weighting == "Equal":
            r_ = g["ret"].mean()
        else:
            w = (w0.reindex(g["CNPJ_FUNDO"]).values if w0 is not None
                 else g["pl_prev"].fillna(0).values)
            r_ = np.average(g["ret"], weights=w) if np.nansum(w) > 0 \
                else g["ret"].mean()
        rows.append({"M": m, "HF": r_, "n_funds": len(g)})

    if cnpj_list:
        sel = df[df["CNPJ_FUNDO"].isin(cnpj_list)]
        for m, g in sel.groupby("M"):
            add_month(m, g)
        for (y, c), g in sel.dropna(subset=["ret"]).groupby(["year", "CNPJ_FUNDO"]):
            picks.append({"year": y, "manager": g["GESTOR"].iat[0],
                          "fund": g["DENOM_SOCIAL"].iat[0], "cnpj": c,
                          "aum_dec_bn": np.nan})
    else:
        for y in years[1:]:
            # Selection uses December of y-1 only (no look-ahead)
            dec = df[df["M"] == pd.Period(f"{y-1}-12", "M")]
            if dec.empty:
                continue
            # Manager AUM: non-FIC funds (masters/standalones), no double count
            aum = (dec[dec["FUNDO_COTAS"] != "S"]
                   .groupby("CPF_CNPJ_GESTOR")["VL_PATRIM_LIQ"].sum()
                   .sort_values(ascending=False))
            names = dec.groupby("CPF_CNPJ_GESTOR")["GESTOR"].agg(
                lambda s: s.mode().iat[0] if not s.mode().empty else "")
            mgrs = list(aum.index[:top_n])
            # Flagship: the manager's largest FIC (the feeder investors buy);
            # if it has none, its largest fund
            cand = dec[dec["CPF_CNPJ_GESTOR"].isin(mgrs)].copy()
            cand["is_fic"] = cand["FUNDO_COTAS"] == "S"
            flag = (cand.sort_values(["is_fic", "VL_PATRIM_LIQ"])
                        .groupby("CPF_CNPJ_GESTOR").tail(1))
            for _, r in flag.iterrows():
                picks.append({"year": y,
                              "manager": names.get(r.CPF_CNPJ_GESTOR, ""),
                              "fund": r.DENOM_SOCIAL, "cnpj": r.CNPJ_FUNDO,
                              "aum_dec_bn": aum.get(r.CPF_CNPJ_GESTOR, np.nan) / 1e9})
            w0 = flag.set_index("CNPJ_FUNDO")["VL_PATRIM_LIQ"]
            yr = df[(df["year"] == y) & df["CNPJ_FUNDO"].isin(set(flag["CNPJ_FUNDO"]))]
            for m, g in yr.groupby("M"):
                add_month(m, g, w0)

    sleeve = pd.DataFrame(rows).set_index("M").sort_index()
    sleeve.index = sleeve.index.to_timestamp(how="end").normalize()
    last_seen = df.dropna(subset=["ret"]).groupby("CNPJ_FUNDO")["M"].max()
    return sleeve, pd.DataFrame(picks), last_seen


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
# App
# ----------------------------------------------------------------------------
def _pct(x):
    return f"{x:.1%}"


def _sig(p):
    return "significant at 5%" if p < 0.05 else (
        "significant at 10%" if p < 0.10 else "not statistically significant")


def main():
    st.set_page_config("BR hedge funds vs Sharpe", layout="wide")
    st.title("Do large-manager Brazilian hedge funds improve Sharpe?")

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

        st.header("Hedge-fund universe")
        top_n = st.slider("Top N managers by multimercado AUM", 5, 40, 15)
        raw_cnpjs = st.text_area(
            "…or fund CNPJs (one per line, overrides Top N)",
            placeholder="22.187.946/0001-41\n12.808.980/0001-32")
        cnpjs = [re.sub(r"\D", "", c).zfill(14)
                 for c in raw_cnpjs.splitlines() if re.sub(r"\D", "", c)]
        weighting = st.radio("Sleeve weighting", ["Equal", "AUM"])
        with st.expander("Advanced"):
            max_abs = st.number_input(
                "Data-error filter: ignore fund monthly returns beyond ±",
                0.1, 5.0, 0.5, step=0.1,
                help="CVM quota series sometimes contain errors (wrong decimal, "
                     "quota splits/consolidations). A month where a fund's "
                     "quota moves more than this (0.5 = ±50%) is treated as "
                     "bad data and excluded for that fund.")

        st.header("Baseline portfolio (weights)")
        w_cdi = st.number_input("CDI", 0.0, 1.0, 0.40)
        w_imab = st.number_input("IMA-B", 0.0, 1.0, 0.20)
        w_ibov = st.number_input("Ibovespa", 0.0, 1.0, 0.20)
        w_world = st.number_input("World equity (BRL)", 0.0, 1.0, 0.20)
        sleeves = st.multiselect("HF sleeve sizes", [0.05, 0.1, 0.2, 0.3, 0.4],
                                 [0.1, 0.2, 0.3])

        st.header("Data sources")
        cdi_code = st.text_input("SGS: CDI monthly %", "4391")
        imab_code = st.text_input("SGS: IMA-B index", "12466")
        ptax_code = st.text_input("SGS: PTAX USD/BRL", "1")
        ibov_tk = st.text_input("Yahoo: Ibovespa", "^BVSP")
        world_tk = st.text_input("Yahoo: World ETF (USD)", "URTH")
        st.caption("Optional CSV overrides (date,value). Levels or returns: "
                   "values > 2 are treated as index levels.")
        up = {k: st.file_uploader(k, type="csv", key=k)
              for k in ["IMA-B", "Ibovespa", "World (BRL)"]}

        run = st.button("Run analysis", type="primary")

    if not run:
        st.info("Set parameters and click **Run analysis**. The first run "
                "downloads several GB of CVM files and can take a while; "
                "later runs use ./data_cache.")
        return
    if (end_m - start_m).n < 24:
        st.error("Use a window of at least 24 months.")
        return
    if not sleeves:
        st.error("Pick at least one HF sleeve size.")
        return

    start = (start_m - 1).to_timestamp().strftime("%Y-%m-%d")
    end = end_m.to_timestamp(how="end").strftime("%Y-%m-%d")

    # --- Registry & universe ------------------------------------------------
    with st.spinner("Loading CVM fund registry…"):
        reg = load_registry()
    mm = reg[reg["CLASSE"].fillna("").str.contains("Multimercado", case=False)]
    mm = mm[mm["FUNDO_EXCLUSIVO"] != "S"]
    if cnpjs:
        unknown = [c for c in cnpjs if c not in set(reg["CNPJ_FUNDO"])]
        if unknown:
            st.warning("Not in the CVM registry (names will be blank, data "
                       f"still loaded if reported): {', '.join(unknown)}")
        universe = frozenset(cnpjs)
        meta = reg
    else:
        universe = frozenset(mm["CNPJ_FUNDO"])
        meta = mm
        st.write(f"Eligible multimercado funds in registry (incl. closed or "
                 f"cancelled): **{len(mm):,}**")

    prog = st.progress(0.0)
    panel = load_fund_panel((start_m - 1).year, end_m.to_timestamp(how="end"),
                            universe, prog)
    prog.empty()
    if panel.empty:
        st.error("No CVM data found for this universe.")
        return

    sleeve, picks, last_seen = build_sleeve(panel, meta, top_n, cnpjs,
                                            weighting, max_abs)

    # --- Benchmarks ---------------------------------------------------------
    with st.spinner("Loading BCB / market data…"):
        cdi = sgs(cdi_code, start, end) / 100
        cdi.index = cdi.index.to_period("M").to_timestamp(how="end").normalize()

        def level_or_upload(key, fetch):
            if up[key] is not None:
                s = read_upload(up[key])
                return to_monthly_returns(s) if s.abs().median() > 2 else \
                    s.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            return fetch()

        imab = level_or_upload(
            "IMA-B", lambda: to_monthly_returns(sgs(imab_code, start, end)))
        ibov = level_or_upload(
            "Ibovespa", lambda: yahoo_monthly_close(ibov_tk, start, end).pct_change())
        ptax = sgs(ptax_code, start, end).resample("ME").last()

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

    base_w = {"CDI": w_cdi, "IMA-B": w_imab, "Ibovespa": w_ibov,
              "World": w_world}
    ports = {"Baseline": portfolio(R, base_w)}
    for s in sorted(sleeves):
        w = {k: v * (1 - s) for k, v in base_w.items()}
        w["HF"] = s
        ports[f"Baseline + {int(s*100)}% HF"] = portfolio(R, w)
    P = pd.DataFrame(ports)
    last = P.columns[-1]
    ex = R.sub(rf, axis=0).drop(columns="CDI")
    risky = ["IMA-B", "Ibovespa", "World"]

    tabs = st.tabs(["Summary", "Sharpe tests", "Spanning & factors",
                    "Robustness", "Universe"])
    fmt = {"Ann. return": "{:.2%}", "Ann. vol": "{:.2%}", "Sharpe": "{:.2f}",
           "Sortino": "{:.2f}", "Max DD": "{:.1%}", "Months": "{:.0f}"}

    # --- Summary ------------------------------------------------------------
    with tabs[0]:
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
        st.subheader("Cumulative excess return over CDI")
        st.line_chart((1 + P.sub(rf, axis=0)).cumprod())

    # --- Sharpe tests -------------------------------------------------------
    with tabs[1]:
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

    # --- Spanning / factors ---------------------------------------------------
    with tabs[2]:
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
    with tabs[3]:
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
        st.line_chart(roll.mean() / roll.std() * np.sqrt(12))
        st.subheader("Rolling 36-month HF alpha (ann.)")
        st.line_chart(alphas)
        st.subheader("Drawdowns")
        wealth = (1 + P).cumprod()
        st.area_chart(wealth / wealth.cummax() - 1)
        st.subheader("Returns in Ibovespa's worst months")
        st.write(pd.DataFrame({
            "Avg return, worst 10% Ibov months": R[worst][["Ibovespa", "HF"]].mean(),
            "Avg return, other months": R[~worst][["Ibovespa", "HF"]].mean(),
        }).style.format("{:.2%}"))

    # --- Universe -------------------------------------------------------------
    with tabs[4]:
        held = picks["cnpj"].unique() if len(picks) else []
        dead = [c for c in held if last_seen.get(c, end_m) < end_m]
        st.info(
            f"The sleeve held **{sleeve['n_funds'].mean():.1f} funds** on average "
            f"over {len(R)} months, using {len(held)} distinct funds. "
            f"**{len(dead)}** of them stopped reporting before the end of the "
            f"sample (closed, merged or migrated) and are kept until their last "
            f"month, so results aren't flattered by survivorship.")
        st.subheader("Funds held each year")
        st.dataframe(picks, width="stretch")
        st.subheader("Number of live funds in sleeve")
        st.line_chart(sleeve["n_funds"])
        st.download_button("Download monthly returns (CSV)",
                           pd.concat([R, P], axis=1).to_csv(),
                           "monthly_returns.csv")


if __name__ == "__main__":
    main()
