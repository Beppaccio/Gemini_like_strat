# gemini.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# =========================
# CONFIGURAZIONE BASE
# =========================

st.set_page_config(page_title="Gemini-like Momentum Screener & Backtest", layout="wide")


@st.cache_data
def get_sp500_tickers():
    """
    Legge la lista completa S&P 500 da un CSV locale (sp500.csv).
    Il file deve essere nella stessa cartella di questo script.
    Deve contenere una colonna 'Symbol' o 'Ticker'.[web:68]
    """
    df = pd.read_csv("sp500.csv")
    col = "Symbol" if "Symbol" in df.columns else "Ticker"
    tickers = df[col].astype(str).tolist()
    # Converte i punti in "-" per compatibilità Yahoo (es. BRK.B -> BRK-B)
    tickers = [t.replace(".", "-").strip() for t in tickers]
    return tickers


@st.cache_data
def download_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    return data


@st.cache_data
def download_volume(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Volume"]
    return data


# =========================
# FILTRI
# =========================

def apply_filters(price_df, vol_df, params, ref_date):
    """
    price_df, vol_df: DataFrame (Date x Ticker)
    ref_date: ultima data disponibile <= data di ribilanciamento
    params: dict con soglie filtri
    """
    # Usa solo dati fino a ref_date
    p = price_df.loc[:ref_date]
    v = vol_df.loc[:ref_date]

    # Servono almeno 252 barre
    min_len = 252
    valid_tickers = [t for t in p.columns if p[t].dropna().shape[0] >= min_len]
    p = p[valid_tickers]
    v = v[valid_tickers]

    if p.empty:
        return []

    # Prezzo medio 20 giorni
    p20 = p.tail(20).mean()
    mask_price = p20 >= params["min_price"]

    # Volume medio 60 giorni
    v60 = v.tail(60).mean()
    mask_vol = v60 >= params["min_volume"]

    # Controvalore medio 60 giorni
    p60 = p.tail(60).mean()
    dollar60 = p60 * v60
    mask_dollar = dollar60 >= params["min_dollar"]

    # SMA200 e trend
    sma200 = p.rolling(200).mean().iloc[-1]
    last_price = p.iloc[-1]
    mask_trend = last_price > sma200

    # Drawdown 6m
    lookback_6m = 126
    if p.shape[0] >= lookback_6m:
        window_6m = p.tail(lookback_6m)
        max_6m = window_6m.max()
        dd_6m = (last_price - max_6m) / max_6m
        mask_dd = dd_6m >= -params["max_dd_6m"]
    else:
        mask_dd = pd.Series(True, index=p.columns)

    # Volatilità 3m
    lookback_3m = 63
    if p.shape[0] >= lookback_3m:
        ret_3m = p.tail(lookback_3m).pct_change().dropna()
        vol_3m = ret_3m.std() * np.sqrt(252)
        perc = np.percentile(vol_3m.dropna(), params["vol_percentile"])
        mask_vol_extreme = vol_3m <= perc
    else:
        mask_vol_extreme = pd.Series(True, index=p.columns)

    mask_all = (
        mask_price &
        mask_vol &
        mask_dollar &
        mask_trend &
        mask_dd &
        mask_vol_extreme
    )

    selected = list(mask_all[mask_all].index)
    return selected


# =========================
# FATTORI MOMENTUM
# =========================

def zscore(series):
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return series * 0
    return (series - mu) / sigma


def compute_momentum_scores(price_df, benchmark_series, selected_tickers, ref_date, factor_weights):
    """
    Calcolo 7 fattori momentum e score composito per i titoli selezionati in ref_date.
    """
    p = price_df.loc[:ref_date, selected_tickers]
    bm = benchmark_series.loc[:ref_date]

    if p.shape[0] < 252:
        return pd.DataFrame(columns=["score"])

    # Rendimenti cumulati
    def cumret(window):
        return p.pct_change().tail(window + 1).add(1).prod() - 1

    r_1m = cumret(21)
    r_3m = cumret(63)
    r_6m = cumret(126)
    r_12m = cumret(252)

    # Escludi ultimo mese dal 12m
    r_12m_ex1m = (1 + r_12m) / (1 + r_1m) - 1

    # Rendimento benchmark 6m e 12m
    rb_6m = bm.pct_change().tail(126 + 1).add(1).prod() - 1
    rb_12m = bm.pct_change().tail(252 + 1).add(1).prod() - 1

    r_6m_exmkt = r_6m - rb_6m

    # Vol‑adjusted 12m
    ret_daily_12m = p.pct_change().tail(252).dropna()
    vol_12m = ret_daily_12m.std() * np.sqrt(252)
    vol_adj = r_12m / vol_12m

    # Z-scores cross‑sectional
    z_1m = zscore(r_1m)
    z_3m = zscore(r_3m)
    z_6m = zscore(r_6m)
    z_12m = zscore(r_12m_ex1m)
    z_mkt = zscore(r_6m_exmkt)
    z_voladj = zscore(vol_adj)
    z_12m_raw = zscore(r_12m)

    df = pd.DataFrame({
        "z_1m": z_1m,
        "z_3m": z_3m,
        "z_6m_raw": z_6m,
        "z_6m_rel": z_mkt,
        "z_12m": z_12m,
        "z_12m_raw": z_12m_raw,
        "z_voladj": z_voladj
    })

    score = (
        factor_weights["z_1m"] * df["z_1m"] +
        factor_weights["z_3m"] * df["z_3m"] +
        factor_weights["z_6m_raw"] * df["z_6m_raw"] +
        factor_weights["z_6m_rel"] * df["z_6m_rel"] +
        factor_weights["z_12m"] * df["z_12m"] +
        factor_weights["z_12m_raw"] * df["z_12m_raw"] +
        factor_weights["z_voladj"] * df["z_voladj"]
    )

    df["score"] = score
    return df.sort_values("score", ascending=False)


# =========================
# REGIME FILTER
# =========================

def compute_regime(benchmark_series, ref_date, ma_window):
    sub = benchmark_series.loc[:ref_date]
    if sub.shape[0] < ma_window:
        return True  # default risk-on se poco storico
    ma = sub.rolling(ma_window).mean()
    price = sub.iloc[-1]
    ma_last = ma.iloc[-1]
    ma_prev = ma.iloc[-ma_window] if ma.shape[0] > ma_window else ma.iloc[0]
    slope = ma_last - ma_prev
    risk_on = (price > ma_last) and (slope >= 0)
    return risk_on


# =========================
# BACKTEST
# =========================

def run_backtest(price_df, vol_df, benchmark_series, params, factor_weights):
    """
    Backtest con ribilanciamento discreto:
    - Mensile: ultimo giorno di ogni mese
    - Settimanale: ogni venerdì
    - Giornaliero: ogni giorno di trading
    """
    all_dates = price_df.index

    if params["rebalance_freq"] == "Mensile":
        rebal_dates = all_dates.to_series().resample("M").last().dropna().tolist()
    elif params["rebalance_freq"] == "Settimanale":
        rebal_dates = all_dates.to_series().resample("W-FRI").last().dropna().tolist()
    else:  # Giornaliero
        rebal_dates = list(all_dates)

    rebal_dates = [d for d in rebal_dates if d >= params["start"] and d <= params["end"]]

    port_val = pd.Series(index=all_dates, dtype=float)
    port_val.iloc[0] = 1.0
    current_weights = pd.Series(dtype=float)
    last_rebal_date = None

    composition_history = []

    for i, dt in enumerate(all_dates):
        if last_rebal_date is None:
            last_rebal_date = dt

        # Ribilanciamento
        if dt in rebal_dates:
            ref_date = all_dates[all_dates <= dt][-1]
            selected = apply_filters(price_df, vol_df, params, ref_date)
            if len(selected) > 0:
                scores_df = compute_momentum_scores(price_df, benchmark_series, selected, ref_date, factor_weights)
                if scores_df.shape[0] > 0:
                    top = scores_df.head(params["n_stocks"])
                    tickers = top.index.tolist()

                    if params["weight_scheme"] == "Equal":
                        base_weights = pd.Series(1.0 / len(tickers), index=tickers)
                    else:
                        sc = top["score"].clip(lower=0)
                        if sc.sum() == 0:
                            base_weights = pd.Series(1.0 / len(tickers), index=tickers)
                        else:
                            base_weights = sc / sc.sum()

                    risk_on = compute_regime(benchmark_series, ref_date, params["ma_window"])
                    exposure = 1.0
                    if not risk_on:
                        exposure = params["risk_off_exposure"]

                    current_weights = base_weights * exposure

                    composition_history.append({
                        "date": dt,
                        "risk_on": risk_on,
                        "exposure": exposure,
                        "weights": current_weights.to_dict()
                    })
                else:
                    current_weights = pd.Series(dtype=float)
            else:
                current_weights = pd.Series(dtype=float)

            last_rebal_date = dt

        # Rendimento giornaliero
        if dt == all_dates[0]:
            continue
        prev_dt = all_dates[all_dates < dt][-1]

        if current_weights.empty:
            ret_day = 0.0
        else:
            prices_today = price_df.loc[dt, current_weights.index]
            prices_prev = price_df.loc[prev_dt, current_weights.index]
            rets = (prices_today / prices_prev) - 1.0
            ret_day = (rets * current_weights).sum()

        port_val.loc[dt] = port_val.loc[prev_dt] * (1.0 + ret_day)

    port_val = port_val.dropna()
    return port_val, composition_history


# =========================
# METRICHE
# =========================

def compute_metrics(equity, benchmark):
    equity = equity.dropna()
    benchmark = benchmark.reindex(equity.index).dropna()
    equity = equity.reindex(benchmark.index)

    rets = equity.pct_change().dropna()
    rets_bm = benchmark.pct_change().dropna()
    rets, rets_bm = rets.align(rets_bm, join="inner")

    if rets.empty:
        return {}

    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = rets.std() * np.sqrt(252)
    sharpe = cagr / vol if vol != 0 else 0

    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = dd.min()

    cov = np.cov(rets, rets_bm)[0, 1]
    var_bm = np.var(rets_bm)
    beta = cov / var_bm if var_bm != 0 else 0
    alpha = (cagr - (rets_bm.mean() * 252)) - 0

    corr = np.corrcoef(rets, rets_bm)[0, 1] if len(rets) > 1 else 0

    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "Alpha": alpha,
        "Beta": beta,
        "Corr": corr
    }


# =========================
# UI STREAMLIT
# =========================

def main():
    st.title("Gemini-like Momentum Screener & Backtest (S&P 500 completo)")

    st.sidebar.header("Parametri Backtest")

    today = datetime.today().date()
    default_start = today - timedelta(days=365 * 5)

    start = st.sidebar.date_input("Data inizio", default_start)
    end = st.sidebar.date_input("Data fine", today)

    universe_choice = st.sidebar.selectbox(
        "Universe",
        ["S&P500 (completo)"]
    )

    min_price = st.sidebar.number_input("Prezzo minimo (media 20 gg)", 0.0, 1000.0, 5.0, 0.5)
    min_volume = st.sidebar.number_input("Volume medio minimo (60 gg)", 0.0, 10_000_000.0, 300_000.0, 50_000.0)
    min_dollar = st.sidebar.number_input("Controvalore medio minimo (USD, 60 gg)", 0.0, 100_000_000.0, 2_000_000.0, 500_000.0)
    max_dd_6m = st.sidebar.number_input("Drawdown massimo 6M (%)", 0.0, 100.0, 35.0, 1.0) / 100.0
    vol_percentile = st.sidebar.number_input("Percentile max volatilità 3M", 80.0, 100.0, 90.0, 1.0)

    n_stocks = st.sidebar.number_input("Numero titoli in portafoglio", 1, 50, 10, 1)
    weight_scheme = st.sidebar.selectbox("Schema pesi", ["Equal", "Proporzionale al momentum"])

    benchmark_choice = st.sidebar.selectbox(
        "Indice di riferimento",
        ["SPY", "QQQ", "VTI", "VOO", "IVV", "IWM"]
    )

    ma_window = st.sidebar.number_input("Periodo MA regime (giorni)", 50, 300, 200, 10)
    risk_off_mode = st.sidebar.selectbox("Comportamento in risk-off", ["50% esposizione", "Cash"])
    risk_off_exposure = 0.5 if risk_off_mode == "50% esposizione" else 0.0

    rebalance_freq = st.sidebar.selectbox(
        "Frequenza ribilanciamento",
        ["Mensile", "Settimanale", "Giornaliero"]
    )

    st.sidebar.header("Pesi fattori (7)")
    fw = {
        "z_1m": st.sidebar.number_input("Peso 1M", -5.0, 5.0, 1.0, 0.1),
        "z_3m": st.sidebar.number_input("Peso 3M", -5.0, 5.0, 1.0, 0.1),
        "z_6m_raw": st.sidebar.number_input("Peso 6M raw", -5.0, 5.0, 1.0, 0.1),
        "z_6m_rel": st.sidebar.number_input("Peso 6M rel vs mkt", -5.0, 5.0, 1.0, 0.1),
        "z_12m": st.sidebar.number_input("Peso 12M (ex 1M)", -5.0, 5.0, 1.0, 0.1),
        "z_12m_raw": st.sidebar.number_input("Peso 12M raw", -5.0, 5.0, 1.0, 0.1),
        "z_voladj": st.sidebar.number_input("Peso vol-adjusted", -5.0, 5.0, 1.0, 0.1),
    }

    params = {
        "start": pd.to_datetime(start),
        "end": pd.to_datetime(end),
        "min_price": min_price,
        "min_volume": min_volume,
        "min_dollar": min_dollar,
        "max_dd_6m": max_dd_6m,
        "vol_percentile": vol_percentile,
        "n_stocks": int(n_stocks),
        "weight_scheme": weight_scheme,
        "ma_window": int(ma_window),
        "risk_off_exposure": risk_off_exposure,
        "rebalance_freq": rebalance_freq
    }

    # =========================
    # DATI
    # =========================

    if universe_choice == "S&P500 (completo)":
        tickers = get_sp500_tickers()
    else:
        tickers = get_sp500_tickers()

    st.write(f"Universe: {len(tickers)} titoli S&P 500 (da sp500.csv).")

    all_tickers = tickers + [benchmark_choice]

    price_df = download_prices(all_tickers, params["start"] - timedelta(days=400), params["end"])
    vol_df = download_volume(all_tickers, params["start"] - timedelta(days=400), params["end"])

    # Split benchmark / titoli
    bm_series = price_df[benchmark_choice]
    price_df = price_df[tickers]
    vol_df = vol_df[tickers]

    tab1, tab2 = st.tabs(["Backtest", "Screener (oggi)"])

    with tab1:
        st.subheader("Risultati Backtest")

        if st.button("Esegui Backtest"):
            equity, comp_hist = run_backtest(price_df, vol_df, bm_series, params, fw)

            if equity.dropna().shape[0] < 10:
                st.warning("Dati insufficienti per un backtest significativo.")
            else:
                bm_aligned = bm_series.reindex(equity.index).dropna()
                equity = equity.reindex(bm_aligned.index)
                bm_norm = bm_aligned / bm_aligned.iloc[0]

                df_eq = pd.DataFrame({
                    "Strategy": equity,
                    benchmark_choice: bm_norm
                })

                st.line_chart(df_eq)

                metrics = compute_metrics(equity, bm_norm)
                st.write("Metriche:")
                st.write({k: round(v, 4) for k, v in metrics.items()})

                if comp_hist:
                    last_n = min(10, len(comp_hist))
                    st.write(f"Ultimi {last_n} ribilanciamenti:")
                    rows = []
                    for ch in comp_hist[-last_n:]:
                        for t, w in ch["weights"].items():
                            rows.append({
                                "date": ch["date"],
                                "ticker": t,
                                "weight": w,
                                "risk_on": ch["risk_on"],
                                "exposure": ch["exposure"]
                            })
                    st.dataframe(pd.DataFrame(rows))

    with tab2:
        st.subheader("Screener su data finale")
        if len(price_df.index) == 0:
            st.warning("Nessun dato di prezzo disponibile.")
        else:
            ref_date = price_df.index[-1]
            selected = apply_filters(price_df, vol_df, params, ref_date)
            st.write(f"Titoli che passano i filtri ({len(selected)}):")
            st.write(selected)

            if selected:
                scores_df = compute_momentum_scores(price_df, bm_series, selected, ref_date, fw)
                st.write("Top titoli per momentum score:")
                st.dataframe(scores_df.head(50))


if __name__ == "__main__":
    main()
