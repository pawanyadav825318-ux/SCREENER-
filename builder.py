"""Chartink-style scan builder: groups (AND/OR), multi-timeframe (Daily/Weekly/Monthly), crossover, offsets."""
import operator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datasource import IST, OHLCV, apply_overlay, fetch_fyers, fetch_quotes, fetch_yahoo, fy_client, throttle
from tech import adx_dmi, atr, rsi, stochastic, vwap
MTF_DAYS = 900  # ~2.5 saal daily history (weekly/monthly resample ke liye)
TF_RULE = {"Chart timeframe": "chart", "Daily": None, "Weekly": "W-FRI", "Monthly": "ME"}
@st.cache_data(ttl=3600, show_spinner=False)
def load_frames(symbols, tf, source, token, hist_bucket, quote_bucket, live):
    """Chart timeframe ke candles (live LTP ke saath). Returns (frames, failed, err, n_quotes)."""
    if source == "fyers":
        frames, failed, err = fetch_fyers(symbols, tf, token, hist_bucket)
        quotes = fetch_quotes(tuple(frames), token, quote_bucket) if (live and frames) else {}
    else:
        frames, failed, err = fetch_yahoo(symbols, tf, hist_bucket)
        quotes = {}
    return {s: apply_overlay(d, quotes.get(s), tf) for s, d in frames.items()}, failed, err, len(quotes)
@st.cache_data(ttl=3600, show_spinner=False)
def load_daily_long(symbols, source, token, bucket):
    """Lambi daily history — Daily/Weekly/Monthly conditions (multi-timeframe) resample karne ke liye."""
    if source == "fyers":
        client = fy_client(token)
        today = datetime.now(IST).date()
        frm, to = (today - timedelta(days=MTF_DAYS)).isoformat(), today.isoformat()
        def one(sym):
            throttle()
            r = client.history(data={"symbol": f"NSE:{sym}-EQ", "resolution": "D", "date_format": "1",
                                     "range_from": frm, "range_to": to, "cont_flag": "1"})
            if r.get("s") != "ok" or not r.get("candles"):
                raise RuntimeError("no data")
            d = pd.DataFrame(r["candles"], columns=["ts"] + OHLCV)
            d.index = pd.DatetimeIndex(pd.to_datetime(d.pop("ts"), unit="s", utc=True)).tz_convert(IST)
            d = d[~d.index.duplicated()].astype(float)
            if len(d) < 30:
                raise RuntimeError("too few bars")
            return d
        frames = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {s: ex.submit(one, s) for s in symbols}
            for s, f in futs.items():
                try:
                    frames[s] = f.result()
                except Exception:
                    pass
        return frames
    tickers = [s + ".NS" for s in symbols]
    raw = yf.download(tickers, period="3y", interval="1d", group_by="ticker",
                      threads=True, progress=False, auto_adjust=False)
    frames = {}
    for s, t in zip(symbols, tickers):
        try:
            d = raw[t] if len(tickers) > 1 else raw
            d = d.rename(columns=str.lower).dropna(subset=["close"])[OHLCV].astype(float)
            if len(d) >= 30:
                frames[s] = d
        except Exception:
            pass
    return frames
def resample_ohlc(daily, rule):
    if rule is None:
        return daily
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return daily.resample(rule).agg(agg).dropna(subset=["close"])
# ---------------- Indicator registry ----------------
def _supertrend(d, p, f, intra):
    mult = float(p["mult"])
    hl2, a, close = (d["high"] + d["low"]) / 2, atr(d, int(p["period"])), d["close"].values
    ub, lb = (hl2 + mult * a).values, (hl2 - mult * a).values
    fub, flb, dirn, line = ub.copy(), lb.copy(), np.ones(len(d)), lb.copy()
    for i in range(1, len(d)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or close[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or close[i - 1] < flb[i - 1]) else flb[i - 1]
        if dirn[i - 1] == 1 and close[i] < flb[i]:
            dirn[i] = -1
        elif dirn[i - 1] == -1 and close[i] > fub[i]:
            dirn[i] = 1
        else:
            dirn[i] = dirn[i - 1]
        line[i] = flb[i] if dirn[i] == 1 else fub[i]
    return pd.Series(dirn if f.startswith("Direction") else line, index=d.index)
def _macd(d, p, f, intra):
    c = d["close"]
    m = c.ewm(span=int(p["fast"]), adjust=False).mean() - c.ewm(span=int(p["slow"]), adjust=False).mean()
    s = m.ewm(span=int(p["signal"]), adjust=False).mean()
    return {"Line": m, "Signal": s, "Histogram": m - s}[f]
def _bb(d, p, f, intra):
    n, k = int(p["period"]), float(p["std"])
    mid, sd = d["close"].rolling(n).mean(), d["close"].rolling(n).std()
    return {"Upper": mid + k * sd, "Middle": mid, "Lower": mid - k * sd}[f]
def _stoch(d, p, f, intra):
    kk, dd = stochastic(d, int(p["k"]), int(p["smooth"]), int(p["d"]))
    return kk if f == "%K" else dd
def _adx(d, p, f, intra):
    a, pdi, mdi = adx_dmi(d, int(p["period"]))
    return {"ADX": a, "+DI": pdi, "-DI": mdi}[f]
def _cci(d, p, f, intra):
    n = int(p["period"])
    tp = (d["high"] + d["low"] + d["close"]) / 3
    dev = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - tp.rolling(n).mean()) / (0.015 * dev)
def _willr(d, p, f, intra):
    n = int(p["period"])
    hh, ll = d["high"].rolling(n).max(), d["low"].rolling(n).min()
    return -100 * (hh - d["close"]) / (hh - ll).replace(0, np.nan)
def _mfi(d, p, f, intra):
    n = int(p["period"])
    tp = (d["high"] + d["low"] + d["close"]) / 3
    mf = tp * d["volume"]
    pos = mf.where(tp > tp.shift(), 0.0).rolling(n).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(n).sum()
    return (100 - 100 / (1 + pos / neg.replace(0, np.nan))).where(neg != 0, 100.0)
def _col(name):
    return lambda d, p, f, intra: d[name]
IND = {  # name: (params [(id, label, default)], fields, function)
    "Close": ([], None, _col("close")),
    "Open": ([], None, _col("open")),
    "High": ([], None, _col("high")),
    "Low": ([], None, _col("low")),
    "Volume": ([], None, _col("volume")),
    "% Change": ([], None, lambda d, p, f, i: d["close"].pct_change() * 100),
    "SMA": ([("period", "Period", 20)], None, lambda d, p, f, i: d["close"].rolling(int(p["period"])).mean()),
    "EMA": ([("period", "Period", 20)], None, lambda d, p, f, i: d["close"].ewm(span=int(p["period"]), adjust=False).mean()),
    "RSI": ([("period", "Period", 14)], None, lambda d, p, f, i: rsi(d["close"], int(p["period"]))),
    "MACD": ([("fast", "Fast", 12), ("slow", "Slow", 26), ("signal", "Signal", 9)],
             ["Line", "Signal", "Histogram"], _macd),
    "Bollinger Bands": ([("period", "Period", 20), ("std", "Std dev", 2.0)], ["Upper", "Middle", "Lower"], _bb),
    "Supertrend": ([("period", "Period", 10), ("mult", "Multiplier", 3.0)],
                   ["Value", "Direction (1=Buy, -1=Sell)"], _supertrend),
    "Stochastic": ([("k", "%K", 14), ("smooth", "Smooth", 3), ("d", "%D", 3)], ["%K", "%D"], _stoch),
    "ADX / DMI": ([("period", "Period", 14)], ["ADX", "+DI", "-DI"], _adx),
    "ATR": ([("period", "Period", 14)], None, lambda d, p, f, i: atr(d, int(p["period"]))),
    "VWAP": ([], None, lambda d, p, f, i: vwap(d, i)),
    "CCI": ([("period", "Period", 20)], None, _cci),
    "Williams %R": ([("period", "Period", 14)], None, _willr),
    "MFI": ([("period", "Period", 14)], None, _mfi),
    "ROC": ([("period", "Period", 12)], None, lambda d, p, f, i: (d["close"] / d["close"].shift(int(p["period"])) - 1) * 100),
    "OBV": ([], None, lambda d, p, f, i: (np.sign(d["close"].diff()).fillna(0) * d["volume"]).cumsum()),
    "Volume SMA": ([("period", "Period", 20)], None, lambda d, p, f, i: d["volume"].rolling(int(p["period"])).mean()),
    "Highest High (pichhle N)": ([("period", "N candles", 20)], None,
                                 lambda d, p, f, i: d["high"].rolling(int(p["period"])).max().shift()),
    "Lowest Low (pichhle N)": ([("period", "N candles", 20)], None,
                               lambda d, p, f, i: d["low"].rolling(int(p["period"])).min().shift()),
}
OPS = [">", "<", ">=", "<=", "==", "crosses above", "crosses below"]
CMP = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le, "==": operator.eq}
OFFSETS = ["Latest"] + [f"{i} candle pehle" for i in range(1, 11)]
DEFAULT = {0: ("RSI", "<", "Number"), 1: ("Close", ">", "EMA")}
PDEF = {"g0_c1R_p_period": 200, "g0_c0L_p_period": 14}
# ---------------- Text helpers ----------------
def op_text(o, mult=True):
    if "num" in o:
        return f"{o['num']:g}"
    s = o["ind"]
    if o["params"]:
        s += "(" + ",".join(f"{v:g}" for v in o["params"].values()) + ")"
    if o["field"]:
        s += " " + o["field"].split(" (")[0]
    if o["off"]:
        s += f" [{o['off']} candle pehle]"
    if mult and o.get("mult", 1.0) != 1.0:
        s = f"{o['mult']:g} x {s}"
    return s
def describe_clause(c):
    tf = c.get("tf", "Chart timeframe")
    pre = "" if tf == "Chart timeframe" else f"[{tf}] "
    t = f"{pre}{op_text(c['left'])} {c['op']} {op_text(c['right'])}"
    if c["within"] > 1:
        t += f"  (pichhle {c['within']} candles mein kabhi bhi)"
    return t
def describe_group(g, n, numbered):
    j = g["join"].split()[0]
    body = f"\n  {j} ".join(describe_clause(c) for c in g["clauses"]) or "(koi condition nahi)"
    body = "  " + body.replace("\n", "\n  ")
    return f"Group {n}:\n{body}" if numbered else body.strip()
def describe(groups, top_join):
    if not groups:
        return "(koi condition nahi)"
    if len(groups) == 1:
        return describe_group(groups[0], 1, numbered=False)
    j = top_join.split()[0]
    return f"\n{j} ".join(describe_group(g, i + 1, numbered=True) for i, g in enumerate(groups))
# ---------------- UI ----------------
def operand_ui(key, side, default_name):
    names = (["Number"] if side == "right" else []) + list(IND)
    name = st.selectbox("Indicator" if side == "left" else "Isse compare karo", names,
                        index=names.index(default_name), key=f"{key}_n")
    if name == "Number":
        return {"num": float(st.number_input("Value", value=30.0, key=f"{key}_v"))}
    plist, fields, _ = IND[name]
    params = {}
    if plist:
        for col, (pid, label, dflt) in zip(st.columns(len(plist)), plist):
            dflt = PDEF.get(f"{key}_p_{pid}", dflt)
            if isinstance(dflt, int):
                params[pid] = col.number_input(label, min_value=1, value=dflt, step=1, key=f"{key}_p_{pid}")
            else:
                params[pid] = col.number_input(label, min_value=0.1, value=float(dflt), step=0.5, key=f"{key}_p_{pid}")
    field = st.selectbox("Line", fields, key=f"{key}_f") if fields else None
    off = OFFSETS.index(st.selectbox("Candle", OFFSETS, key=f"{key}_o"))
    mult = 1.0
    if side == "right":
        mult = float(st.number_input("x Multiplier (jaise 1.5 x)", value=1.0, step=0.05, key=f"{key}_m"))
    return {"ind": name, "params": params, "field": field, "off": off, "mult": mult}
def clause_ui(prefix, cid, n):
    deleted = False
    with st.container(border=True):
        head = st.columns([5, 1])
        head[0].markdown(f"Condition {n}")
        if head[1].button("🗑️", key=f"{prefix}del{cid}"):
            deleted = True
        tf = st.selectbox("Timeframe (Chartink ke Monthly/Weekly/Daily jaisa)", list(TF_RULE),
                          key=f"{prefix}tf{cid}")
        ln, dop, rn = DEFAULT.get(cid, ("Close", ">", "Number"))
        left = operand_ui(f"{prefix}c{cid}L", "left", ln)
        op = st.selectbox("Operator", OPS, index=OPS.index(dop), key=f"{prefix}op{cid}")
        right = operand_ui(f"{prefix}c{cid}R", "right", rn)
        within = int(st.number_input("Pichhle kitne candles mein? (1 = sirf latest)", 1, 20, 1, key=f"{prefix}w{cid}"))
    return {"left": left, "op": op, "right": right, "within": within, "tf": tf}, deleted
def group_ui(gid, n, allow_delete):
    ss = st.session_state
    ss.setdefault(f"g{gid}_ids", [0, 1] if gid == 0 else [0])
    ss.setdefault(f"g{gid}_next", 2 if gid == 0 else 1)
    del_group = False
    with st.container(border=True):
        top = st.columns([5, 1])
        top[0].markdown(f"### 📦 Group {n}")
        if allow_delete and top[1].button("🗑️ Group", key=f"gdel{gid}"):
            del_group = True
        join = st.radio("Is group ke andar", ["AND (sabhi sach ho)", "OR (koi ek sach ho)"],
                        horizontal=True, key=f"g{gid}join")
        clauses = []
        for i, cid in enumerate(list(ss[f"g{gid}_ids"]), 1):
            c, deleted = clause_ui(f"g{gid}_", cid, i)
            if deleted and len(ss[f"g{gid}_ids"]) > 1:
                ss[f"g{gid}_ids"].remove(cid)
                st.rerun()
            clauses.append(c)
        if st.button("➕ Condition jodo", key=f"g{gid}addc"):
            ss[f"g{gid}_ids"].append(ss[f"g{gid}_next"])
            ss[f"g{gid}_next"] += 1
            st.rerun()
    return {"clauses": clauses, "join": join}, del_group
def builder_ui():
    ss = st.session_state
    ss.setdefault("grp_ids", [0])
    ss.setdefault("grp_next", 1)
    groups = []
    for n, gid in enumerate(list(ss["grp_ids"]), 1):
        g, del_group = group_ui(gid, n, allow_delete=len(ss["grp_ids"]) > 1)
        if del_group:
            ss["grp_ids"].remove(gid)
            st.rerun()
        groups.append(g)
    top_join = "OR (koi ek group sach ho)"
    if len(groups) > 1:
        top_join = st.radio("Groups ko kaise jodein?",
                            ["OR (koi ek group sach ho)", "AND (sabhi groups sach ho)"],
                            horizontal=True, key="grp_join")
    if st.button("➕ Naya Group jodo (Chartink ke 'block' jaisa)"):
        ss["grp_ids"].append(ss["grp_next"])
        ss["grp_next"] += 1
        st.rerun()
    return groups, top_join
# ---------------- Engine ----------------
def _get_df(sym, tf, chart_frames, daily_frames):
    if tf == "Chart timeframe":
        return chart_frames.get(sym)
    daily = daily_frames.get(sym)
    if daily is None:
        return None
    return resample_ohlc(daily, TF_RULE[tf])
def _series(df, o, intraday, cache, tf):
    key = (tf, o["ind"], tuple(o["params"].items()), o["field"])
    if key not in cache:
        cache[key] = IND[o["ind"]][2](df, o["params"], o["field"], intraday)
    return cache[key]
def _val(sym, o, back, cache, chart_frames, daily_frames, chart_intraday, tf, mult=True):
    if "num" in o:
        return o["num"]
    df = _get_df(sym, tf, chart_frames, daily_frames)
    if df is None or len(df) < 5:
        return np.nan
    is_intra = chart_intraday if tf == "Chart timeframe" else False
    s = _series(df, o, is_intra, cache, tf)
    i = -1 - o["off"] - back
    if abs(i) > len(s):
        return np.nan
    v = s.iloc[i]
    return v * o["mult"] if mult else v
def _clause_ok(sym, c, cache, chart_frames, daily_frames, chart_intraday):
    tf = c.get("tf", "Chart timeframe")
    for back in range(c["within"]):
        a = _val(sym, c["left"], back, cache, chart_frames, daily_frames, chart_intraday, tf)
        b = _val(sym, c["right"], back, cache, chart_frames, daily_frames, chart_intraday, tf)
        if c["op"] in ("crosses above", "crosses below"):
            a0 = _val(sym, c["left"], back + 1, cache, chart_frames, daily_frames, chart_intraday, tf)
            b0 = _val(sym, c["right"], back + 1, cache, chart_frames, daily_frames, chart_intraday, tf)
            ok = (a0 <= b0 and a > b) if c["op"] == "crosses above" else (a0 >= b0 and a < b)
        else:
            ok = CMP[c["op"]](a, b)
        if ok:
            return True
    return False
def _group_ok(sym, g, cache, chart_frames, daily_frames, chart_intraday):
    if not g["clauses"]:
        return True
    res = [_clause_ok(sym, c, cache, chart_frames, daily_frames, chart_intraday) for c in g["clauses"]]
    return all(res) if g["join"].startswith("AND") else any(res)
def run_scan(chart_frames, daily_frames, groups, top_join, chart_intraday):
    rows = {}
    for sym, df in chart_frames.items():
        try:
            cache = {}
            gres = [_group_ok(sym, g, cache, chart_frames, daily_frames, chart_intraday) for g in groups]
            passed = all(gres) if top_join.startswith("AND") else any(gres)
            if not passed:
                continue
            close = df["close"]
            row = {"close": close.iloc[-1], "chg_pct": (close.iloc[-1] / close.iloc[-2] - 1) * 100,
                   "volume": df["volume"].iloc[-1]}
            for g in groups:
                for c in g["clauses"]:
                    tf = c.get("tf", "Chart timeframe")
                    for o in (c["left"], c["right"]):
                        if "ind" in o:
                            label = op_text(o, mult=False) + ("" if tf == "Chart timeframe" else f" [{tf}]")
                            row[label] = _val(sym, o, 0, cache, chart_frames, daily_frames, chart_intraday, tf, mult=False)
            rows[sym] = row
        except Exception:
            continue
    return pd.DataFrame(rows).T.astype(float) if rows else pd.DataFrame()
