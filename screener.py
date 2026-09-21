"""
Pro Technical Screener (Chartink-style) - NSE, Fyers real-time
UI file. Data: datasource.py | Indicators: tech.py
"""
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from datasource import (HIST_TTL, IST, TFS, apply_overlay, build, fetch_fyers, fetch_quotes, fetch_yahoo,
                        fyers_configured, fyers_session, fyers_token, market_is_open, secret, token_store)
from tech import indicators

st.set_page_config(page_title="Pro Technical Screener", page_icon="📈", layout="wide")

NIFTY50 = """ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJFINANCE BAJAJFINSV BPCL BHARTIARTL
BRITANNIA CIPLA COALINDIA DIVISLAB DRREDDY EICHERMOT GRASIM HCLTECH HDFCBANK HDFCLIFE HEROMOTOCO HINDALCO
HINDUNILVR ICICIBANK ITC INDUSINDBK INFY JSWSTEEL KOTAKBANK LT M&M MARUTI NTPC NESTLEIND ONGC POWERGRID
RELIANCE SBILIFE SHRIRAMFIN SBIN SUNPHARMA TCS TATACONSUM TMPV TATASTEEL TECHM TITAN ULTRACEMCO WIPRO""".split()

EXTRA = """ADANIGREEN ADANIPOWER AMBUJACEM BANKBARODA BEL CANBK CHOLAFIN DLF DABUR GAIL GODREJCP HAL HAVELLS
ICICIGI IOC IRCTC JINDALSTEL LICI LUPIN MUTHOOTFIN NAUKRI PFC PIDILITIND PNB RECLTD SIEMENS TVSMOTOR
TRENT VBL VEDL ETERNAL ZYDUSLIFE TMCV""".split()

PRESETS = {
    "RSI Oversold (RSI14 < 30)": "rsi14 < 30",
    "RSI Overbought (RSI14 > 70)": "rsi14 > 70",
    "RSI crosses above 50": "prev_rsi14 <= 50 and rsi14 > 50",
    "EMA 9/20 Bullish Crossover": "prev_ema9 <= prev_ema20 and ema9 > ema20",
    "EMA 20/50 Bullish Crossover": "prev_ema20 <= prev_ema50 and ema20 > ema50",
    "EMA 20/50 Bearish Crossover": "prev_ema20 >= prev_ema50 and ema20 < ema50",
    "Price above EMA200 + RSI > 55": "close > ema200 and rsi14 > 55",
    "Golden Cross (SMA50 > SMA200)": "prev_sma50 <= prev_sma200 and sma50 > sma200",
    "Death Cross (SMA50 < SMA200)": "prev_sma50 >= prev_sma200 and sma50 < sma200",
    "MACD Bullish Crossover": "prev_macd <= prev_macd_sig and macd > macd_sig",
    "MACD Bearish Crossover": "prev_macd >= prev_macd_sig and macd < macd_sig",
    "Close above Bollinger Upper": "close > bb_up",
    "Close below Bollinger Lower": "close < bb_lo",
    "Bollinger Squeeze (width < 4%)": "bb_width < 4",
    "BB Lower + RSI < 35 (reversal)": "close <= bb_lo and rsi14 < 35",
    "Supertrend Buy (fresh)": "st_dir == 1 and prev_st_dir == -1",
    "Supertrend Sell (fresh)": "st_dir == -1 and prev_st_dir == 1",
    "Stochastic Oversold Crossover": "prev_stoch_k <= prev_stoch_d and stoch_k > stoch_d and stoch_k < 25",
    "Stochastic Overbought Crossover": "prev_stoch_k >= prev_stoch_d and stoch_k < stoch_d and stoch_k > 75",
    "ADX Strong Uptrend (ADX>25, +DI>-DI)": "adx > 25 and plus_di > minus_di",
    "ADX Strong Downtrend (ADX>25, -DI>+DI)": "adx > 25 and minus_di > plus_di",
    "DI Bullish Crossover": "prev_plus_di <= prev_minus_di and plus_di > minus_di",
    "Price crosses above VWAP": "prev_close <= prev_vwap and close > vwap",
    "Price crosses below VWAP": "prev_close >= prev_vwap and close < vwap",
    "High Volatility (ATR% > 3)": "atr_pct > 3",
    "Volume Breakout (Vol > 2x avg, green)": "vol_ratio > 2 and close > open",
    "20-Candle High Breakout": "close > high20",
    "20-Candle Low Breakdown": "close < low20",
}

COLS = ["close", "open", "volume", "chg_pct", "rsi14", "ema9", "ema20", "ema50", "ema200", "sma20", "sma50",
        "sma200", "macd", "macd_sig", "macd_hist", "bb_up", "bb_lo", "bb_width", "bb_pct", "st_dir", "stoch_k",
        "stoch_d", "adx", "plus_di", "minus_di", "vwap", "atr14", "atr_pct", "vol_ratio", "high20", "low20"]
OPS = [">", "<", ">=", "<=", "crosses above", "crosses below"]
SHOW = ["close", "chg_pct", "rsi14", "adx", "stoch_k", "bb_pct", "vwap", "atr_pct", "vol_ratio", "st_dir",
        "ema20", "ema50", "ema200"]


def make_chart(o, symbol, intraday):
    o = o.tail(150)
    x = o.index.strftime("%d %b %H:%M" if intraday else "%d %b %y")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.5, 0.15, 0.17, 0.18])
    fig.add_trace(go.Candlestick(x=x, open=o.open, high=o.high, low=o.low, close=o.close, name=symbol), 1, 1)
    for col, color in [("ema20", "#f5a623"), ("ema50", "#4a90e2"), ("ema200", "#9b59b6")]:
        fig.add_trace(go.Scatter(x=x, y=o[col], name=col.upper(), line=dict(width=1.2, color=color)), 1, 1)
    for col in ("bb_up", "bb_lo"):
        fig.add_trace(go.Scatter(x=x, y=o[col], name=col, line=dict(width=1, dash="dot", color="gray")), 1, 1)
    if intraday:
        fig.add_trace(go.Scatter(x=x, y=o.vwap, name="VWAP", line=dict(width=1.2, color="#e74c3c")), 1, 1)
    fig.add_trace(go.Bar(x=x, y=o.volume, name="Volume", marker_color="#95a5a6"), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=o.rsi14, name="RSI", line=dict(color="#8e44ad")), 3, 1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_trace(go.Bar(x=x, y=o.macd_hist, name="MACD hist", marker_color="#bdc3c7"), 4, 1)
    fig.add_trace(go.Scatter(x=x, y=o.macd, name="MACD", line=dict(width=1, color="#2980b9")), 4, 1)
    fig.add_trace(go.Scatter(x=x, y=o.macd_sig, name="Signal", line=dict(width=1, color="#e67e22")), 4, 1)
    fig.update_xaxes(type="category", nticks=8, rangeslider_visible=False)
    fig.update_layout(height=760, margin=dict(l=5, r=5, t=30, b=5), legend=dict(orientation="h", y=1.04),
                      title=f"{symbol} • {len(o)} candles")
    return fig


def visual_builder():
    n = st.number_input("Kitni conditions?", 1, 6, 2)
    join = st.radio("Combine karo", ["AND", "OR"], horizontal=True)
    defaults = ["rsi14", "close", "adx", "vol_ratio", "macd", "stoch_k"]
    parts = []
    for i in range(n):
        a, b, c_, d = st.columns([3, 3, 2, 3])
        left = a.selectbox("Indicator", COLS, index=COLS.index(defaults[i]), key=f"L{i}", label_visibility="collapsed")
        op = b.selectbox("Operator", OPS, key=f"op{i}", label_visibility="collapsed")
        mode = c_.selectbox("vs", ["Value", "Indicator"], key=f"m{i}", label_visibility="collapsed")
        if mode == "Value":
            v = d.number_input("Value", value=50.0, key=f"v{i}", label_visibility="collapsed")
            r, pr = repr(float(v)), repr(float(v))
        else:
            r = d.selectbox("Indicator 2", COLS, index=COLS.index("ema200"), key=f"R{i}", label_visibility="collapsed")
            pr = "prev_" + r
        if op == "crosses above":
            parts.append(f"(prev_{left} <= {pr} and {left} > {r})")
        elif op == "crosses below":
            parts.append(f"(prev_{left} >= {pr} and {left} < {r})")
        else:
            parts.append(f"({left} {op} {r})")
    return f" {join.lower()} ".join(parts)


# ======================= App =======================
FYERS_CONFIGURED = fyers_configured()
now = datetime.now(IST)
market_open = market_is_open(now)

pw = secret("APP_PASSWORD")
if pw and not st.session_state.get("authed"):
    st.title("🔒 Pro Technical Screener")
    typed = st.text_input("Password", type="password")
    if typed and typed == pw:
        st.session_state["authed"] = True
        st.rerun()
    elif typed:
        st.error("Galat password")
    st.stop()

with st.sidebar:
    st.title("⚙️ Settings")
    st.caption(f"{'🟢 NSE OPEN' if market_open else '🔴 NSE CLOSED'} • {now:%d %b %H:%M} IST")
    src_label = st.radio("Data source", ["Fyers (real-time)", "Yahoo (15 min delay)"],
                         index=0 if FYERS_CONFIGURED else 1)
    source = "fyers" if src_label.startswith("Fyers") else "yahoo"
    universe = st.selectbox("Stock list", ["Nifty 50", "Nifty 50 + Extra (~90)", "Custom"])
    if universe == "Custom":
        txt = st.text_area("NSE symbols (comma separated)", "RELIANCE, TCS, INFY, SBIN")
        symbols = tuple(dict.fromkeys(x.strip().upper() for x in txt.split(",") if x.strip()))
    else:
        symbols = tuple(NIFTY50 if universe == "Nifty 50" else NIFTY50 + EXTRA)
    tf = st.selectbox("Timeframe", TFS, index=1)
    live = st.checkbox("Live LTP overlay", value=True, disabled=source != "fyers",
                       help="Chalti candle par abhi ka live price lagata hai")
    if st.button("🔄 Refresh data now", width="stretch"):
        st.cache_data.clear()
    auto = st.checkbox("Auto refresh", value=False)
    every = st.slider("Every (sec)", 10, 300, 15, 5, disabled=not auto)
    if source == "fyers" and FYERS_CONFIGURED and st.button("🔐 Fyers dobara login", width="stretch"):
        token_store().clear()
        st.rerun()

st.title("📈 Pro Technical Screener")

token = ""
if source == "fyers":
    if not FYERS_CONFIGURED:
        st.warning("Fyers setup baaki hai. Streamlit **Settings → Secrets** mein FYERS_APP_ID, FYERS_SECRET, "
                   "FYERS_REDIRECT daalo (`secrets_example.toml` dekho). Tab tak sidebar mein Yahoo chuno.")
        st.stop()
    token = fyers_token()
    if not token:
        st.info("Fyers se aaj ka login karna hoga (din mein ek baar).")
        st.link_button("🔐 Fyers se Login", fyers_session().generate_authcode(), type="primary")
        st.caption("Login ke baad app apne aap khul jaayegi. Ya sidebar se Yahoo chuno.")
        st.stop()

if not symbols:
    st.warning("Sidebar mein kam se kam ek symbol daalein.")
    st.stop()

now_ts = time.time()
hb = int(now_ts // (HIST_TTL[tf] if source == "fyers" else 60))
qb = int(now_ts // 10)
with st.spinner("Data la raha hoon... (pehli baar 10-20 sec lag sakte hain)"):
    data, failed, err, nq = build(symbols, tf, source, token, hb, qb, live)

if data.empty:
    st.error(f"Data nahi mila. {('API: ' + err) if err else ''}")
    if source == "fyers":
        st.caption("Token expire ho gaya ho sakta hai — sidebar mein 'Fyers dobara login' dabao.")
    st.stop()

tag = f"Fyers live ({nq} LTP)" if source == "fyers" and nq else ("Fyers" if source == "fyers" else "Yahoo delayed")
st.caption(f"{len(data)} stocks • {tf} • {tag} • updated {datetime.now(IST):%H:%M:%S}"
           + (f" • ⚠️ {len(failed)} load nahi hue" if failed else ""))
if failed:
    with st.expander("Load na hone wale symbols"):
        st.write(", ".join(failed) + " — symbol rename/delist ho sakta hai." + (f" API: {err}" if err else ""))

tab_scan, tab_dash, tab_chart = st.tabs(["🔍 Scanner", "📊 Dashboard", "🕯️ Chart"])

with tab_scan:
    mode = st.radio("Scan kaise banayein?", ["Ready scans", "Visual builder", "Type condition"], horizontal=True)
    if mode == "Ready scans":
        query = PRESETS[st.selectbox("Scan", list(PRESETS))]
    elif mode == "Visual builder":
        query = visual_builder()
    else:
        query = st.text_area("Condition", "close > ema200 and rsi14 > 60 and adx > 25 and vol_ratio > 1.5",
                             help="Pichla candle: prev_ lagao, jaise prev_rsi14")
    st.code(query or "(no condition)", language="python")
    try:
        res = data.query(query, engine="python") if query.strip() else data
    except Exception as e:
        st.error(f"Condition mein error: {e}")
        res = None
    if res is not None:
        st.subheader(f"✅ {len(res)} stocks mile")
        if len(res):
            table = res[SHOW].round(2).sort_values("chg_pct", ascending=False)
            st.dataframe(table, width="stretch")
            st.download_button("⬇️ CSV download", table.to_csv().encode(), "scan_results.csv", "text/csv")
            st.markdown("TradingView: " + " • ".join(
                f"[{s}](https://www.tradingview.com/chart/?symbol=NSE:{s.replace('&', '_').replace('-', '_')})"
                for s in res.index[:15]))
        else:
            st.info("Abhi koi stock is condition par match nahi karta.")
    with st.expander("📚 Available indicators / columns"):
        st.markdown("""
| # | Indicator | Columns |
|---|---|---|
| 1 | **RSI (14)** | `rsi14` |
| 2 | **EMA** | `ema9` `ema20` `ema50` `ema200` |
| 3 | **SMA** | `sma20` `sma50` `sma200` |
| 4 | **MACD (12,26,9)** | `macd` `macd_sig` `macd_hist` |
| 5 | **Bollinger (20,2)** | `bb_up` `bb_lo` `bb_width` `bb_pct` |
| 6 | **Supertrend (10,3)** | `st_dir` (1 buy, -1 sell) |
| 7 | **Stochastic (14,3,3)** | `stoch_k` `stoch_d` |
| 8 | **ADX / DMI (14)** | `adx` `plus_di` `minus_di` |
| 9 | **VWAP** | `vwap` (intraday session / 1d = 20-day) |
| 10 | **ATR (14)** | `atr14` `atr_pct` |
| + | Price / Volume | `open` `close` `volume` `chg_pct` `vol_ratio` `high20` `low20` |

Pichla candle: `prev_` lagao (`prev_rsi14`, `prev_close`).
""")

with tab_dash:
    st.subheader("Saare scans ek saath")
    rows = []
    for name, q in PRESETS.items():
        try:
            r = data.query(q, engine="python")
        except Exception:
            continue
        rows.append({"Scan": name, "Count": len(r),
                     "Stocks": ", ".join(r.index[:12]) + (" ..." if len(r) > 12 else "")})
    st.dataframe(pd.DataFrame(rows).sort_values("Count", ascending=False), width="stretch", hide_index=True)
    up = int((data["chg_pct"] > 0).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Advancing", up)
    c2.metric("Declining", len(data) - up)
    c3.metric("Avg RSI", f"{data['rsi14'].mean():.1f}")

with tab_chart:
    pick = st.selectbox("Stock", list(data.index))
    try:
        if source == "fyers":
            frames, _, _ = fetch_fyers(symbols, tf, token, hb)
            q = fetch_quotes(tuple(frames), token, qb) if live else {}
        else:
            frames, _, _ = fetch_yahoo(symbols, tf, hb)
            q = {}
        df = apply_overlay(frames[pick], q.get(pick), tf)
        st.plotly_chart(make_chart(indicators(df, tf != "1d"), pick, tf != "1d"), width="stretch")
    except Exception as e:
        st.error(f"Chart load nahi hua: {e}")

if auto:
    time.sleep(every)
    st.rerun()
