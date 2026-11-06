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
def build_sleeve(panel, reg, top_n, min_cotst, manual_managers,
                 weighting, max_abs_ret):
    df = panel.merge(reg, on="CNPJ_FUNDO", how="left")
    df = df.sort_values(["CNPJ_FUNDO", "M"])
    df["ret"] = df.groupby("CNPJ_FUNDO")["VL_QUOTA"].pct_change()
    # Only accept consecutive months; flag absurd quota jumps
    gap = df.groupby("CNPJ_FUNDO")["M"].diff().apply(
        lambda x: x.n if pd.notna(x) else np.nan)
    df.loc[gap != 1, "ret"] = np.nan
    df.loc[df["ret"].abs() > max_abs_ret, "ret"] = np.nan

    df["year"] = df["M"].dt.year
    months = sorted(df["M"].unique())
    years = sorted({m.year for m in months})
    rows, picks = [], []

    for y in years[1:]:
        # Selection uses December of y-1 only (no look-ahead)
        dec = df[df["M"] == pd.Period(f"{y-1}-12", "M")]
        if dec.empty:
            continue
        # Manager AUM: non-FIC funds (masters/standalones) to avoid double count
        aum = (dec[dec["FUNDO_COTAS"] != "S"]
               .groupby("CPF_CNPJ_GESTOR")["VL_PATRIM_LIQ"].sum()
               .sort_values(ascending=False))
        names = dec.groupby("CPF_CNPJ_GESTOR")["GESTOR"].agg(
            lambda s: s.mode().iat[0] if not s.mode().empty else "")
        if manual_managers:
            pat = "|".join(re.escape(m) for m in manual_managers)
            mgrs = [g for g in aum.index
                    if re.search(pat, str(names.get(g, "")), re.I)]
        else:
            mgrs = list(aum.index[:top_n])

        # Flagship per manager: largest investable fund (many quotaholders,
        # which naturally picks the feeder investors actually buy)
        inv = dec[(dec["NR_COTST"] >= min_cotst)
                  & dec["CPF_CNPJ_GESTOR"].isin(mgrs)]
        flag = (inv.sort_values("VL_PATRIM_LIQ")
                   .groupby("CPF_CNPJ_GESTOR").tail(1))
        for _, r in flag.iterrows():
            picks.append({"year": y, "manager": names.get(r.CPF_CNPJ_GESTOR, ""),
                          "fund": r.DENOM_SOCIAL, "cnpj": r.CNPJ_FUNDO,
                          "aum_dec_bn": aum.get(r.CPF_CNPJ_GESTOR, np.nan) / 1e9})
        sel = set(flag["CNPJ_FUNDO"])
        w0 = flag.set_index("CNPJ_FUNDO")["VL_PATRIM_LIQ"]

        yr = df[(df["year"] == y) & df["CNPJ_FUNDO"].isin(sel)]
        for m, g in yr.groupby("M"):
            g = g.dropna(subset=["ret"])  # dead funds drop out when they stop
            if g.empty:
                continue
            if weighting == "Equal":
                r_ = g["ret"].mean()
            else:
                w = w0.reindex(g["CNPJ_FUNDO"]).values
                r_ = np.average(g["ret"], weights=w)
            rows.append({"M": m, "HF": r_, "n_funds": len(g)})

    sleeve = pd.DataFrame(rows).set_index("M").sort_index()
    sleeve.index = sleeve.index.to_timestamp(how="end").normalize()
    return sleeve, pd.DataFrame(picks)


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
def main():
    st.set_page_config("BR hedge funds vs Sharpe", layout="wide")
    st.title("Do large-manager Brazilian hedge funds improve Sharpe?")

    with st.sidebar:
        st.header("Period")
        start_year = st.number_input("Start year", 2005, 2025, 2010)
        end_date = st.date_input("End date", date.today())

        st.header("Hedge-fund universe")
        top_n = st.slider("Top N managers by multimercado AUM", 5, 40, 15)
        manual = st.text_area(
            "…or manager names (one per line, overrides Top N)",
            placeholder="VERDE\nSPX\nKAPITALO\nLEGACY\nIBIUNA")
        manual = [m.strip() for m in manual.splitlines() if m.strip()]
        min_cotst = st.number_input("Min. quotaholders (investable fund)", 1,
                                    100000, 100)
        excl_prev = st.checkbox("Exclude previdência (name contains PREV)", True)
        excl_qual = st.checkbox("Exclude qualified-investor-only funds", False)
        weighting = st.radio("Sleeve weighting", ["Equal", "AUM"])
        max_abs = st.number_input("Drop monthly returns above (abs)", 0.1,
                                  5.0, 0.5)

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
        use_nefin = st.checkbox("Add NEFIN factors to factor regression", True)

        run = st.button("Run analysis", type="primary")

    if not run:
        st.info("Set parameters and click **Run analysis**. The first run "
                "downloads several GB of CVM files and can take a while; "
                "later runs use ./data_cache.")
        return

    start = f"{start_year - 1}-12-01"
    end = str(end_date)

    # --- Registry & universe ------------------------------------------------
    with st.spinner("Loading CVM fund registry…"):
        reg = load_registry()
    mm = reg[reg["CLASSE"].fillna("").str.contains("Multimercado", case=False)]
    mm = mm[mm["FUNDO_EXCLUSIVO"] != "S"]
    if excl_prev:
        mm = mm[~mm["DENOM_SOCIAL"].fillna("").str.contains("PREV", case=False)]
    if excl_qual and "INVEST_QUALIF" in mm:
        mm = mm[mm["INVEST_QUALIF"] != "S"]
    st.write(f"Multimercado funds in registry (incl. closed/cancelled): "
             f"**{len(mm):,}**")

    prog = st.progress(0.0)
    panel = load_fund_panel(start_year - 1, pd.Timestamp(end_date),
                            frozenset(mm["CNPJ_FUNDO"]), prog)
    prog.empty()

    sleeve, picks = build_sleeve(panel, mm, top_n, min_cotst, manual,
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
    R = R.loc[f"{start_year}-01-01":].dropna()
    rf = R["CDI"]
    st.caption(f"Sample: {R.index[0]:%b %Y} – {R.index[-1]:%b %Y} "
               f"({len(R)} months)")

    base_w = {"CDI": w_cdi, "IMA-B": w_imab, "Ibovespa": w_ibov,
              "World": w_world}
    ports = {"Baseline": portfolio(R, base_w)}
    for s in sleeves:
        w = {k: v * (1 - s) for k, v in base_w.items()}
        w["HF"] = s
        ports[f"Baseline + {int(s*100)}% HF"] = portfolio(R, w)
    P = pd.DataFrame(ports)

    tabs = st.tabs(["Summary", "Sharpe tests", "Spanning & factors",
                    "Robustness", "Universe"])

    # --- Summary ------------------------------------------------------------
    with tabs[0]:
        stats = pd.DataFrame({k: ann_stats(P[k], rf) for k in P}).T
        assets = pd.DataFrame({k: ann_stats(R[k], rf)
                               for k in ["IMA-B", "Ibovespa", "World", "HF"]}).T
        st.subheader("Portfolios")
        st.dataframe(stats.style.format({
            "Ann. return": "{:.2%}", "Ann. vol": "{:.2%}", "Sharpe": "{:.2f}",
            "Sortino": "{:.2f}", "Max DD": "{:.1%}", "Months": "{:.0f}"}))
        st.subheader("Building blocks")
        st.dataframe(assets.style.format({
            "Ann. return": "{:.2%}", "Ann. vol": "{:.2%}", "Sharpe": "{:.2f}",
            "Sortino": "{:.2f}", "Max DD": "{:.1%}", "Months": "{:.0f}"}))
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
        st.markdown("Ledoit-Wolf (2008) HAC test, H0: Sharpe equal to baseline.")
        st.dataframe(pd.DataFrame(rows).set_index("Portfolio").style.format(
            {"ΔSharpe (ann.)": "{:+.3f}", "SE (ann.)": "{:.3f}",
             "p-value": "{:.3f}"}))

    # --- Spanning / factors ---------------------------------------------------
    with tabs[2]:
        ex = R.sub(rf, axis=0).drop(columns="CDI")
        risky = ["IMA-B", "Ibovespa", "World"]
        m1 = hac_ols(ex["HF"], ex[risky])
        a, se_eps = m1.params["const"], np.sqrt(m1.scale)
        sr_base = tangency_sharpe(ex[risky])
        ar = a / se_eps * np.sqrt(12)
        st.subheader("Spanning regression: HF excess on baseline assets")
        st.text(m1.summary().tables[1].as_text())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Alpha (ann.)", f"{a*12:.2%}",
                  f"t = {m1.tvalues['const']:.2f}", delta_color="off")
        c2.metric("Appraisal ratio (ann.)", f"{ar:.2f}")
        c3.metric("Tangency Sharpe, base assets", f"{sr_base:.2f}")
        c4.metric("Tangency Sharpe, + HF", f"{np.sqrt(sr_base**2 + ar**2):.2f}")
        st.caption("Positive, significant alpha ⇒ HF expands the efficient "
                   "frontier: SR²(new) = SR²(old) + (α/σε)². In-sample "
                   "tangency Sharpes are upper bounds, not achievable figures.")

        st.subheader("Factor regression (is the alpha cheap beta?)")
        F = ex[risky].copy()
        F["USD/BRL"] = fx.reindex(F.index)
        if use_nefin:
            nf = nefin_factors()
            if nf.empty:
                st.warning("NEFIN factors unavailable; skipped.")
            else:
                nf = nf.drop(columns="Market", errors="ignore")
                F = F.join(nf, how="left")
        F = F.dropna(axis=1, how="all")
        m2 = hac_ols(ex["HF"], F)
        st.text(m2.summary().tables[1].as_text())
        st.caption(f"R² = {m2.rsquared:.2f}, N = {int(m2.nobs)}. "
                   f"Alpha (ann.) = {m2.params['const']*12:.2%}, "
                   f"t = {m2.tvalues['const']:.2f}")

    # --- Robustness -----------------------------------------------------------
    with tabs[3]:
        st.subheader("Sub-periods")
        cuts = [("2010-01", "2014-12"), ("2015-01", "2019-12"),
                ("2020-01", "2022-12"), ("2023-01", str(end_date)[:7])]
        sub = []
        for s0, s1 in cuts:
            Ps, rfs = P.loc[s0:s1], rf.loc[s0:s1]
            if len(Ps) < 24:
                continue
            row = {"Period": f"{s0[:4]}–{s1[:4]}"}
            for k in P:
                e = Ps[k] - rfs
                row[k] = e.mean() / e.std() * np.sqrt(12)
            k_last = P.columns[-1]
            _, _, p = sharpe_diff_test((Ps[k_last] - rfs).values,
                                       (Ps["Baseline"] - rfs).values)
            row[f"p ({k_last} vs base)"] = p
            mm_ = hac_ols((R["HF"] - rf).loc[s0:s1],
                          ex[risky].loc[s0:s1])
            row["HF alpha ann."] = mm_.params["const"] * 12
            row["alpha t"] = mm_.tvalues["const"]
            sub.append(row)
        st.dataframe(pd.DataFrame(sub).set_index("Period").style.format("{:.2f}"))

        st.subheader("Rolling 36-month Sharpe")
        roll = P.sub(rf, axis=0).rolling(36)
        st.line_chart(roll.mean() / roll.std() * np.sqrt(12))

        st.subheader("Rolling 36-month HF alpha (ann.)")
        alphas = {}
        for i in range(36, len(ex) + 1):
            w_ = ex.iloc[i - 36:i]
            X = sm.add_constant(w_[risky])
            alphas[w_.index[-1]] = sm.OLS(w_["HF"], X).fit().params["const"] * 12
        st.line_chart(pd.Series(alphas, name="alpha"))

        st.subheader("Drawdowns")
        wealth = (1 + P).cumprod()
        st.area_chart(wealth / wealth.cummax() - 1)

        st.subheader("HF correlation with Ibovespa in its worst months")
        worst = R["Ibovespa"] <= R["Ibovespa"].quantile(0.1)
        st.write(pd.DataFrame({
            "Avg return, worst 10% Ibov months": R[worst][["Ibovespa", "HF"]].mean(),
            "Avg return, other months": R[~worst][["Ibovespa", "HF"]].mean(),
        }).style.format("{:.2%}"))

    # --- Universe -------------------------------------------------------------
    with tabs[4]:
        st.subheader("Funds held each year (selected in prior December)")
        st.dataframe(picks, use_container_width=True)
        st.subheader("Number of live funds in sleeve")
        st.line_chart(sleeve["n_funds"])
        st.download_button("Download monthly returns (CSV)",
                           pd.concat([R, P], axis=1).to_csv(),
                           "monthly_returns.csv")


if __name__ == "__main__":
    main()
