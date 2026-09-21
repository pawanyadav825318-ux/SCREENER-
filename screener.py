"""
📈 PRO TECHNICAL SCREENER (Chartink-style) — NSE, Fyers real-time
Data: Fyers API v3 (history candles + live LTP overlay)  |  fallback: Yahoo (15 min delay)
Top 10 indicators: RSI, EMA, SMA, MACD, Bollinger, Supertrend, Stochastic, ADX/DMI, VWAP, ATR
Tabs: Scanner (ready / visual builder / typed) | Dashboard (all scans) | Chart
Sirf market-data endpoints use hote hain — koi order/trade code nahi hai.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="Pro Technical Screener", page_icon="📈", layout="wide")
IST = ZoneInfo("Asia/Kolkata")

NIFTY50 = """ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJFINANCE BAJAJFINSV BPCL BHARTIARTL
BRITANNIA CIPLA COALINDIA DIVISLAB DRREDDY EICHERMOT GRASIM HCLTECH HDFCBANK HDFCLIFE HEROMOTOCO HINDALCO
HINDUNILVR ICICIBANK ITC INDUSINDBK INFY JSWSTEEL KOTAKBANK LT M&M MARUTI NTPC NESTLEIND ONGC POWERGRID
RELIANCE SBILIFE SHRIRAMFIN SBIN SUNPHARMA TCS TATACONSUM TMPV TATASTEEL TECHM TITAN ULTRACEMCO WIPRO""".split()

EXTRA = """ADANIGREEN ADANIPOWER AMBUJACEM BANKBARODA BEL CANBK CHOLAFIN DLF DABUR GAIL GODREJCP HAL HAVELLS
ICICIGI IOC IRCTC JINDALSTEL LICI LUPIN MUTHOOTFIN NAUKRI PFC PIDILITIND PNB RECLTD SIEMENS TVSMOTOR
TRENT VBL VEDL ETERNAL ZYDUSLIFE TMCV""".split()

TFS = ["5m", "15m", "1h", "1d"]
FY_RES = {"5m": ("5", 10), "15m": ("15", 30), "1h": ("60", 90), "1d": ("D", 360)}  # resolution, lookback days
YF_RES = {"5m": ("5m", "5d"), "15m": ("15m", "1mo"), "1h": ("60m", "3mo"), "1d": ("1d", "1y")}
SPAN = {"5m": 300, "15m": 900, "1h": 3600, "1d": 86400}          # candle length (sec)
HIST_TTL = {"5m": 60, "15m": 120, "1h": 300, "1d": 600}          # history refetch interval (sec)
OHLCV = ["open", "high", "low", "close", "volume"]

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


# ======================= Indicators =======================
def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def true_range(df):
    pc = df["close"].shift()
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df, n=14):
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def adx_dmi(df, n=14):
    up, dn = df["high"].diff(), -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def stochastic(df, n=14, k=3, d=3):
    ll, hh = df["low"].rolling(n).min(), df["high"].rolling(n).max()
    raw = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    kk = raw.rolling(k).mean()
    return kk, kk.rolling(d).mean()


def vwap(df, intraday):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv, v = tp * df["volume"], df["volume"]
    if intraday:  # session VWAP, resets daily
        g = np.array(df.index.date)
        return pv.groupby(g).cumsum() / v.groupby(g).cumsum().replace(0, np.nan)
    return pv.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)


def supertrend_dir(df, period=10, mult=3.0):
    hl2 = (df["high"] + df["low"]) / 2
    a = atr(df, period)
    ub, lb = (hl2 + mult * a).values, (hl2 - mult * a).values
    close = df["close"].values
    fub, flb = ub.copy(), lb.copy()
    d = np.ones(len(df))
    for i in range(1, len(df)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or close[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or close[i - 1] < flb[i - 1]) else flb[i - 1]
        if d[i - 1] == 1 and close[i] < flb[i]:
            d[i] = -1
        elif d[i - 1] == -1 and close[i] > fub[i]:
            d[i] = 1
        else:
            d[i] = d[i - 1]
    return pd.Series(d, index=df.index)


def indicators(df, intraday=False):
    c = df["close"]
    o = pd.DataFrame(index=df.index)
    o["open"], o["high"], o["low"], o["close"], o["volume"] = df["open"], df["high"], df["low"], c, df["volume"]
    o["chg_pct"] = c.pct_change() * 100
    for n in (9, 20, 50, 200):
        o[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    for n in (20, 50, 200):
        o[f"sma{n}"] = c.rolling(n).mean()
    o["rsi14"] = rsi(c)
    o["macd"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    o["macd_sig"] = o["macd"].ewm(span=9, adjust=False).mean()
    o["macd_hist"] = o["macd"] - o["macd_sig"]
    sd = c.rolling(20).std()
    o["bb_up"], o["bb_lo"] = o["sma20"] + 2 * sd, o["sma20"] - 2 * sd
    o["bb_width"] = (o["bb_up"] - o["bb_lo"]) / o["sma20"] * 100
    o["bb_pct"] = (c - o["bb_lo"]) / (o["bb_up"] - o["bb_lo"]).replace(0, np.nan)
    o["st_dir"] = supertrend_dir(df)
    o["stoch_k"], o["stoch_d"] = stochastic(df)
    o["adx"], o["plus_di"], o["minus_di"] = adx_dmi(df)
    o["vwap"] = vwap(df, intraday)
    o["atr14"] = atr(df, 14)
    o["atr_pct"] = o["atr14"] / c * 100
    o["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean().shift()
    o["high20"], o["low20"] = df["high"].rolling(20).max().shift(), df["low"].rolling(20).min().shift()
    return o


def snapshot(o):
    row = o.iloc[-1].to_dict()
    row.update({"prev_" + k: v for k, v in o.iloc[-2].items()})
    return row


# ======================= Secrets / helpers =======================
def secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default


APP_ID, APP_SECRET, REDIRECT = secret("FYERS_APP_ID"), secret("FYERS_SECRET"), secret("FYERS_REDIRECT")
FYERS_CONFIGURED = bool(APP_ID and APP_SECRET and REDIRECT)


def market_is_open(now):
    return now.weekday() < 5 and (9, 15) <= (now.hour, now.minute) <= (15, 30)


_rl_lock, _rl_last = threading.Lock(), [0.0]


def throttle(gap=0.12):  # ~8 calls/sec, Fyers limit se neeche
    with _rl_lock:
        wait = gap - (time.time() - _rl_last[0])
        if wait > 0:
            time.sleep(wait)
        _rl_last[0] = time.time()


@st.cache_resource
def token_store():
    return {}


# ======================= Fyers data =======================
def fy_client(token):
    from fyers_apiv3 import fyersModel
    return fyersModel.FyersModel(client_id=APP_ID, token=token, is_async=False, log_path="")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fyers(symbols, tf, token, bucket):
    """Historical candles for all symbols. Returns (frames, failed, first_error)."""
    res, days = FY_RES[tf]
    client = fy_client(token)
    today = datetime.now(IST).date()
    frm, to = (today - timedelta(days=days)).isoformat(), today.isoformat()

    def one(sym):
        r = {}
        for _ in range(2):
            throttle()
            r = client.history(data={"symbol": f"NSE:{sym}-EQ", "resolution": res, "date_format": "1",
                                     "range_from": frm, "range_to": to, "cont_flag": "1"})
            if r.get("s") == "ok":
                break
            if r.get("code") == 429 or "limit" in str(r.get("message", "")).lower():
                time.sleep(1.5)
                continue
            break
        if r.get("s") != "ok" or not r.get("candles"):
            raise RuntimeError(str(r.get("message") or r))
        d = pd.DataFrame(r["candles"], columns=["ts"] + OHLCV)
        d.index = pd.DatetimeIndex(pd.to_datetime(d.pop("ts"), unit="s", utc=True)).tz_convert(IST)
        d = d[~d.index.duplicated()].astype(float)
        if len(d) < 30:
            raise RuntimeError("too few bars")
        return d

    frames, failed, err = {}, [], None
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {s: ex.submit(one, s) for s in symbols}
        for s, f in futs.items():
            try:
                frames[s] = f.result()
            except Exception as e:
                failed.append(s)
                err = err or str(e)
    return frames, failed, err


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_quotes(symbols, token, bucket):
    """Live LTP etc. (50 symbols per call)."""
    client, out = fy_client(token), {}
    for i in range(0, len(symbols), 50):
        chunk = symbols[i:i + 50]
        throttle()
        r = client.quotes(data={"symbols": ",".join(f"NSE:{s}-EQ" for s in chunk)})
        if r.get("s") != "ok":
            continue
        for d in r.get("d", []):
            v = d.get("v") or {}
            if d.get("s") == "ok" and v.get("lp"):
                out[d["n"].removeprefix("NSE:").removesuffix("-EQ")] = v
    return out


def apply_overlay(df, q, tf):
    """Forming candle par live LTP lagao (agar last candle abhi chal raha hai)."""
    if not q or not q.get("lp"):
        return df
    last = df.index[-1]
    now = datetime.now(IST)
    forming = now.date() == last.date() if tf == "1d" else now < last + timedelta(seconds=SPAN[tf])
    if not forming:
        return df
    lp = float(q["lp"])
    df = df.copy()
    df.loc[last, "close"] = lp
    df.loc[last, "high"] = max(df.loc[last, "high"], lp)
    df.loc[last, "low"] = min(df.loc[last, "low"], lp)
    if tf == "1d" and q.get("volume"):
        df.loc[last, "volume"] = float(q["volume"])
    return df


# ======================= Yahoo fallback =======================
def _extract(raw, t):
    if isinstance(raw.columns, pd.MultiIndex):
        if t in raw.columns.get_level_values(0):
            return raw[t]
        if t in raw.columns.get_level_values(1):
            return raw.xs(t, axis=1, level=1)
        return None
    return raw


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo(symbols, tf, bucket):
    interval, period = YF_RES[tf]
    tickers = [s + ".NS" for s in symbols]
    raw = yf.download(tickers, period=period, interval=interval, group_by="ticker",
                      threads=True, progress=False, auto_adjust=False)
    frames, failed = {}, []
    for s, t in zip(symbols, tickers):
        try:
            d = _extract(raw, t).rename(columns=str.lower).dropna(subset=["close"])[OHLCV].astype(float)
            if len(d) < 30:
                raise ValueError
            frames[s] = d
        except Exception:
            failed.append(s)
    return frames, failed, None


# ======================= Pipeline =======================
@st.cache_data(ttl=3600, show_spinner=False)
def build(symbols, tf, source, token, hist_bucket, quote_bucket, live):
    if source == "fyers":
        frames, failed, err = fetch_fyers(symbols, tf, token, hist_bucket)
        quotes = fetch_quotes(tuple(frames), token, quote_bucket) if (live and frames) else {}
    else:
        frames, failed, err = fetch_yahoo(symbols, tf, hist_bucket)
        quotes = {}
    intraday = tf != "1d"
    rows = {}
    for s, d in frames.items():
        try:
            rows[s] = snapshot(indicators(apply_overlay(d, quotes.get(s), tf), intraday))
        except Exception:
            failed.append(s)
    df = pd.DataFrame(rows).T
    return (df.astype(float) if not df.empty else df), failed, err, len(quotes)


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


# ======================= Fyers login =======================
def fyers_session():
    from fyers_apiv3 import fyersModel
    return fyersModel.SessionModel(client_id=APP_ID, secret_key=APP_SECRET, redirect_uri=REDIRECT,
                                   response_type="code", grant_type="authorization_code", state="screener")


def fyers_token():
    """Valid token (aaj ka) ya None. Login redirect (?auth_code=) ko bhi handle karta hai."""
    store = token_store()
    today = datetime.now(IST).date().isoformat()
    if store.get("token") and store.get("date") == today:
        return store["token"]
    code = st.query_params.get("auth_code")
    if code:
        try:
            sess = fyers_session()
            sess.set_token(code)
            resp = sess.generate_token()
            if resp.get("access_token"):
                store.update(token=resp["access_token"], date=today)
                st.query_params.clear()
                st.rerun()
            st.error(f"Login fail: {resp.get('message') or resp}")
        except st.errors.StreamlitAPIException:
            raise
        except Exception as e:
            if type(e).__name__ in ("RerunException", "StopException"):
                raise
            st.error(f"Login error: {e}")
    return None


# ======================= App =======================
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
    if source == "fyers" and FYER
