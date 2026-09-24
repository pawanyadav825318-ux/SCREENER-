"""Data layer: Fyers real-time (12 timeframes, resample-based) + Yahoo fallback, Fyers login."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import yfinance as yf

from tech import indicators, resample_ohlc, snapshot

IST = ZoneInfo("Asia/Kolkata")
OHLCV = ["open", "high", "low", "close", "volume"]

# Har chart-timeframe ko ek "base" fetch (jo Fyers/Yahoo se seedha milta hai) + resample rule mein todte hain.
# Isse 2h/3h/4h/Weekly/Monthly jaise timeframe bhi bina extra API load ke ban jaate hain.
TFS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "3h", "4h", "1d", "1w", "1M"]
TF_BASE = {"1m": "1m", "3m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h",
          "2h": "1h", "3h": "1h", "4h": "1h", "1d": "1d", "1w": "1d_long", "1M": "1d_long"}
TF_RESAMPLE = {"3m": "3min", "2h": "2h", "3h": "3h", "4h": "4h", "1w": "W-FRI", "1M": "ME"}
SPAN = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200,
       "3h": 10800, "4h": 14400, "1d": 86400, "1w": 604800, "1M": 2592000}
BASE_FY_RES = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "1d": "D", "1d_long": "D"}
BASE_FY_DAYS = {"1m": 5, "5m": 15, "15m": 30, "30m": 45, "1h": 100, "1d": 365, "1d_long": 900}
BASE_YF_INT = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "60m", "1d": "1d", "1d_long": "1d"}
BASE_YF_PERIOD = {"1m": "5d", "5m": "1mo", "15m": "1mo", "30m": "2mo", "1h": "3mo", "1d": "1y", "1d_long": "3y"}
HIST_TTL = {"1m": 30, "3m": 45, "5m": 60, "15m": 120, "30m": 180, "1h": 300, "2h": 400,
           "3h": 500, "4h": 600, "1d": 900, "1w": 1800, "1M": 3600}


# ======================= Secrets / helpers =======================
def secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default


def fy_cfg():
    return secret("FYERS_APP_ID"), secret("FYERS_SECRET"), secret("FYERS_REDIRECT")


def fyers_configured():
    return all(fy_cfg())


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


# ======================= Fyers base fetch =======================
def fy_client(token):
    from fyers_apiv3 import fyersModel
    return fyersModel.FyersModel(client_id=fy_cfg()[0], token=token, is_async=False, log_path="")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fyers_base(symbols, base, token, bucket):
    """Ek 'base' resolution ke candles Fyers se. Returns (frames, failed, first_error)."""
    res, days = BASE_FY_RES[base], BASE_FY_DAYS[base]
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
    if not q or not q.get("lp") or df is None or df.empty:
        return df
    last = df.index[-1]
    now = datetime.now(IST)
    span = SPAN.get(tf, 900)
    forming = now.date() == last.date() if tf in ("1d", "1w", "1M") else now < last + timedelta(seconds=span)
    if not forming:
        return df
    lp = float(q["lp"])
    df = df.copy()
    df.loc[last, "close"] = lp
    df.loc[last, "high"] = max(df.loc[last, "high"], lp)
    df.loc[last, "low"] = min(df.loc[last, "low"], lp)
    if tf in ("1d", "1w", "1M") and q.get("volume"):
        df.loc[last, "volume"] = float(q["volume"])
    return df


# ======================= Yahoo base fetch =======================
def _extract(raw, t):
    if isinstance(raw.columns, pd.MultiIndex):
        if t in raw.columns.get_level_values(0):
            return raw[t]
        if t in raw.columns.get_level_values(1):
            return raw.xs(t, axis=1, level=1)
        return None
    return raw


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_base(symbols, base, bucket):
    interval, period = BASE_YF_INT[base], BASE_YF_PERIOD[base]
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


# ======================= Unified fetch (chart timeframe) =======================
def fetch_ohlc(symbols, tf, source, token, bucket):
    """Kisi bhi chart timeframe (1m..1M) ke candles — base fetch + zaroorat ho to resample."""
    base = TF_BASE[tf]
    if source == "fyers":
        frames, failed, err = fetch_fyers_base(symbols, base, token, bucket)
    else:
        frames, failed, err = fetch_yahoo_base(symbols, base, bucket)
    rule = TF_RESAMPLE.get(tf)
    if rule:
        out = {}
        for s, d in frames.items():
            r = resample_ohlc(d, rule)
            if len(r) >= 20:
                out[s] = r
            else:
                failed.append(s)
        frames = out
    return frames, failed, err


# ======================= Pipeline (Ready scans / Type condition tabs) =======================
@st.cache_data(ttl=3600, show_spinner=False)
def build(symbols, tf, source, token, hist_bucket, quote_bucket, live):
    frames, failed, err = fetch_ohlc(symbols, tf, source, token, hist_bucket)
    quotes = fetch_quotes(tuple(frames), token, quote_bucket) if (source == "fyers" and live and frames) else {}
    intraday = tf not in ("1d", "1w", "1M")
    rows = {}
    for s, d in frames.items():
        try:
            rows[s] = snapshot(indicators(apply_overlay(d, quotes.get(s), tf), intraday))
        except Exception:
            failed.append(s)
    df = pd.DataFrame(rows).T
    return (df.astype(float) if not df.empty else df), failed, err, len(quotes)


# ======================= Fyers login =======================
def fyers_session():
    from fyers_apiv3 import fyersModel
    app_id, secret_key, redirect = fy_cfg()
    return fyersModel.SessionModel(client_id=app_id, secret_key=secret_key, redirect_uri=redirect,
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
