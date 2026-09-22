"""Data layer: Fyers real-time (history + live quotes) with Yahoo fallback, Fyers login."""
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import yfinance as yf

from tech import indicators, snapshot

IST = ZoneInfo("Asia/Kolkata")
TFS = ["5m", "15m", "1h", "1d"]
FY_RES = {"5m": ("5", 10), "15m": ("15", 30), "1h": ("60", 90), "1d": ("D", 360)}  # resolution, default days
FY_STEP = {"5m": 90, "15m": 90, "1h": 90, "1d": 360}  # Fyers: max days per request
YF_RES = {"5m": ("5m", "5d"), "15m": ("15m", "1mo"), "1h": ("60m", "3mo"), "1d": ("1d", "1y")}
SPAN = {"5m": 300, "15m": 900, "1h": 3600, "1d": 86400}  # candle length (sec)
HIST_TTL = {"5m": 300, "15m": 600, "1h": 900, "1d": 21600}  # Fyers history refetch (sec)
OHLCV = ["open", "high", "low", "close", "volume"]


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


def hist_bucket(tf, source):
    return int(time.time() // (HIST_TTL[tf] if source == "fyers" else 60))


_rl_lock, _rl_last = threading.Lock(), [0.0]


def throttle(gap=0.31):  # ~190 calls/min (Fyers limit 200/min se neeche)
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
    return fyersModel.FyersModel(client_id=fy_cfg()[0], token=token, is_async=False, log_path="")


@st.cache_data(ttl=3600, max_entries=6, show_spinner=False)
def fetch_fyers(symbols, tf, token, bucket, days=0):
    """Candles for all symbols (lambi history ke liye chunks mein). Returns (frames, failed, first_error)."""
    res, dflt_days = FY_RES[tf]
    step = FY_STEP[tf]
    client = fy_client(token)
    today = datetime.now(IST).date()
    windows = [((today - timedelta(days=(k + 1) * step)).isoformat(), (today - timedelta(days=k * step)).isoformat())
               for k in range(math.ceil((days or dflt_days) / step))]

    def call(sym, frm, to):
        r = {}
        for _ in range(2):
            throttle()
            r = client.history(data={"symbol": f"NSE:{sym}-EQ", "resolution": res, "date_format": "1",
                                     "range_from": frm, "range_to": to, "cont_flag": "1"})
            if r.get("s") in ("ok", "no_data"):
                break
            if r.get("code") == 429 or "limit" in str(r.get("message", "")).lower():
                time.sleep(2)
                continue
            break
        return r

    def one(sym):
        parts = []
        for i, (frm, to) in enumerate(windows):
            r = call(sym, frm, to)
            if r.get("s") == "ok" and r.get("candles"):
                parts.append(pd.DataFrame(r["candles"], columns=["ts"] + OHLCV))
            elif r.get("s") != "no_data":
                raise RuntimeError(str(r.get("message") or r))
        if not parts:
            raise RuntimeError("no data")
        d = pd.concat(parts)
        d.index = pd.DatetimeIndex(pd.to_datetime(d.pop("ts"), unit="s", utc=True)).tz_convert(IST)
        d = d[~d.index.duplicated()].sort_index().astype(float)
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


@st.cache_data(ttl=3600, max_entries=6, show_spinner=False)
def fetch_quotes(symbols, token, bucket):
    """Live quotes (50 symbols per call): lp, open_price, high_price, low_price, volume..."""
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
    """Market open mein live quote se chalti candle update karo ya nayi candle jodo."""
    if not q or not q.get("lp") or df.empty:
        return df
    now = datetime.now(IST)
    if not market_is_open(now):
        return df
    lp, last, tz = float(q["lp"]), df.index[-1], df.index.tz
    df = df.copy()
    if tf == "1d":
        o = float(q.get("open_price") or lp)
        h, low = max(float(q.get("high_price") or lp), lp), min(float(q.get("low_price") or lp), lp)
        v = float(q.get("volume") or df["volume"].iloc[-1])
        if last.date() == now.date():
            df.loc[last, OHLCV] = [o, h, low, lp, v]
            return df
        idx = pd.Timestamp(now.date())
        idx = idx.tz_localize(tz) if tz is not None else idx
        return pd.concat([df, pd.DataFrame([[o, h, low, lp, v]], columns=OHLCV, index=[idx])])
    span = timedelta(seconds=SPAN[tf])
    if now < last + span:  # last candle abhi chal raha hai
        df.loc[last, "close"] = lp
        df.loc[last, "high"] = max(df.loc[last, "high"], lp)
        df.loc[last, "low"] = min(df.loc[last, "low"], lp)
        return df
    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    slot = open_t + ((now - open_t) // span) * span
    if slot <= last:
        return df
    idx = pd.Timestamp(slot)
    idx = idx.tz_convert(tz) if tz is not None else idx.tz_localize(None)
    return pd.concat([df, pd.DataFrame([[lp, lp, lp, lp, 0.0]], columns=OHLCV, index=[idx])])


def resample_ohlc(df, weekly):
    """Daily candles -> Weekly (W-FRI) ya Monthly."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    try:
        r = df.resample("W-FRI" if weekly else "ME").agg(agg)
    except ValueError:
        r = df.resample("W-FRI" if weekly else "M").agg(agg)
    return r.dropna(subset=["close"])


# ======================= Yahoo fallback =======================
def _extract(raw, t):
    if isinstance(raw.columns, pd.MultiIndex):
        if t in raw.columns.get_level_values(0):
            return raw[t]
        if t in raw.columns.get_level_values(1):
            return raw.xs(t, axis=1, level=1)
        return None
    return raw


@st.cache_data(ttl=3600, max_entries=6, show_spinner=False)
def fetch_yahoo(symbols, tf, bucket, days=0):
    interval, period = YF_RES[tf]
    if tf == "1d" and days > 366:
        period = "5y"
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
@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def load_frames(symbols, tf, source, token, hist_bkt, quote_bkt, live, days=0):
    """Candles + live overlay. Returns (frames, failed, err, n_quotes)."""
    if source == "fyers":
        frames, failed, err = fetch_fyers(symbols, tf, token, hist_bkt, days)
        quotes = fetch_quotes(tuple(frames), token, quote_bkt) if (live and frames) else {}
    else:
        frames, failed, err = fetch_yahoo(symbols, tf, hist_bkt, days)
        quotes = {}
    return {s: apply_overlay(d, quotes.get(s), tf) for s, d in frames.items()}, failed, err, len(quotes)


@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def build(symbols, tf, source, token, hist_bkt, quote_bkt, live):
    """Ready-scan / dashboard ke liye indicator snapshot."""
    frames, failed, err, nq = load_frames(symbols, tf, source, token, hist_bkt, quote_bkt, live)
    failed, rows = list(failed), {}
    for s, d in frames.items():
        try:
            rows[s] = snapshot(indicators(d, tf != "1d"))
        except Exception:
            failed.append(s)
    df = pd.DataFrame(rows).T
    return (df.astype(float) if not df.empty else df), failed, err, nq


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
