"""جلب الشموع من بورصات عامة (بدون مفاتيح API). يجرّب Binance ثم Bybit ثم OKX."""
import time
import numpy as np
import pandas as pd
import requests

TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}
_BYBIT = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60"}
_OKX = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1H"}
COLS = ["time", "open", "high", "low", "close", "volume"]


def to_symbol(pair: str) -> str:
    """BTC/USD (اسم IQ Option) -> BTCUSDT"""
    base = pair.upper().replace(" ", "").split("/")[0]
    return base + "USDT"


def _finish(rows, tf: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLS)
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    for c in COLS[1:]:
        df[c] = df[c].astype(float)
    df = df.drop_duplicates("time").sort_values("time").set_index("time")
    # احذف الشمعة غير المكتملة
    closed = df.index + pd.Timedelta(seconds=TF_SECONDS[tf]) <= pd.Timestamp.now(tz="UTC")
    return df[closed]


def _binance(sym, tf, limit, end=None):
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": sym, "interval": tf, "limit": min(limit, 1000),
                             **({"endTime": end} if end else {})}, timeout=10)
    r.raise_for_status()
    return [[x[0], x[1], x[2], x[3], x[4], x[5]] for x in r.json()]


def _bybit(sym, tf, limit, end=None):
    r = requests.get("https://api.bybit.com/v5/market/kline",
                     params={"category": "spot", "symbol": sym, "interval": _BYBIT[tf],
                             "limit": min(limit, 1000), **({"end": end} if end else {})}, timeout=10)
    r.raise_for_status()
    return [[int(x[0]), x[1], x[2], x[3], x[4], x[5]] for x in r.json()["result"]["list"]]


def _okx(sym, tf, limit, end=None):
    inst = sym.replace("USDT", "-USDT")
    r = requests.get("https://www.okx.com/api/v5/market/candles",
                     params={"instId": inst, "bar": _OKX[tf], "limit": min(limit, 300),
                             **({"after": end} if end else {})}, timeout=10)
    r.raise_for_status()
    return [[int(x[0]), x[1], x[2], x[3], x[4], x[5]] for x in r.json()["data"]]


_SOURCES = [("binance", _binance), ("bybit", _bybit), ("okx", _okx)]


def fetch(pair: str, tf: str, limit: int = 500) -> pd.DataFrame:
    """آخر `limit` شمعة مكتملة. يدعم أكثر من 1000 بالتصفح للخلف (للاختبار التاريخي)."""
    sym, err = to_symbol(pair), []
    for name, fn in _SOURCES:
        try:
            rows, end = [], None
            while len(rows) < limit:
                chunk = fn(sym, tf, limit - len(rows), end)
                if not chunk:
                    break
                rows = chunk + rows if name == "binance" else rows + chunk
                oldest = min(int(c[0]) for c in chunk)
                end = oldest - 1
                if len(chunk) < 2 or name == "okx":
                    break
                time.sleep(0.15)
            if rows:
                return _finish(rows, tf).tail(limit)
        except Exception as e:  # noqa
            err.append(f"{name}: {e}")
    raise RuntimeError(f"تعذّر جلب {pair} من كل المصادر -> " + " | ".join(err))


def synthetic(n=6000, tf="5m", seed=0, trending=True) -> pd.DataFrame:
    """بيانات اصطناعية للاختبار دون إنترنت (مشي عشوائي مع تغيّر في النظام)."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.normal(0, 0.0004 if trending else 0, n // 200 + 1), 200)[:n]
    ret = drift + rng.normal(0, 0.0025, n)
    close = 30000 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(rng.normal(0, 0.0015, (n, 2))) * close[:, None]
    high = np.maximum(open_, close) + wick[:, 0]
    low = np.minimum(open_, close) - wick[:, 1]
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h"), periods=n, freq=f"{TF_SECONDS[tf]}s")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.uniform(1, 10, n)}, index=idx)


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    return df.resample(f"{TF_SECONDS[tf]}s", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
