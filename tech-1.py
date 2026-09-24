"""Technical indicators (RSI, EMA, SMA, MACD, BB, Supertrend, Stochastic, ADX, VWAP, ATR)."""
import numpy as np
import pandas as pd


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

def resample_ohlc(daily, rule):
    """Chhoti candle ki OHLCV ko bade timeframe (Weekly/Monthly) mein badalta hai."""
    if rule is None:
        return daily
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return daily.resample(rule).agg(agg).dropna(subset=["close"])
