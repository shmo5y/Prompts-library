"""
محرّك الإشارات: نقاط من -100 (بيع/Put قوي) إلى +100 (شراء/Call قوي).
نفس الدالة تُستخدم للتحليل المباشر وللاختبار التاريخي، فالنتائج متطابقة ولا يوجد تسرّب مستقبلي.

نظامان:
  1) اتجاه + ارتداد (ADX >= 20): الدخول مع الاتجاه بعد تصحيح بسيط، بشرط توافق الفريم الأكبر.
  2) تذبذب (ADX < 20): دخول عكسي عند تطرّف بولينجر + RSI مع شمعة انعكاس.
فلتر تذبذب (ATR): يمنع الإشارات في السوق الميت أو أثناء الانفجارات السعرية.
"""
import numpy as np
import pandas as pd
from indicators import adx, atr, bollinger_pctb, ema, macd_hist, rsi
from data import TF_SECONDS


def htf_trend(htf: pd.DataFrame, htf_tf: str) -> pd.Series:
    """اتجاه الفريم الأكبر (+1/-1/0) ويُتاح فقط بعد إغلاق شمعته."""
    c = htf["close"]
    e20, e50 = ema(c, 20), ema(c, 50)
    t = pd.Series(np.where((e20 > e50) & (c > e20), 1, np.where((e20 < e50) & (c < e20), -1, 0)), index=htf.index)
    t.index = t.index + pd.Timedelta(seconds=TF_SECONDS[htf_tf])  # وقت الإغلاق = وقت التوفر
    return t


def compute(df: pd.DataFrame, htf: pd.DataFrame, tf: str, htf_tf: str) -> pd.DataFrame:
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    out = pd.DataFrame(index=df.index)
    e9, e21, e50 = ema(c, 9), ema(c, 21), ema(c, 50)
    r = rsi(c)
    mh = macd_hist(c)
    pb = bollinger_pctb(c)
    a = atr(df)
    ax, pdi, mdi = adx(df)
    rng = (h - l).replace(0, np.nan)
    body = (c - o) / rng
    lower_wick = (np.minimum(o, c) - l) / rng
    upper_wick = (h - np.maximum(o, c)) / rng
    vol_ratio = a / a.rolling(100).mean()

    # اتجاه الفريم الأكبر مُحاذى لوقت إغلاق شمعة الإشارة
    ht = htf_trend(htf, htf_tf)
    close_time = pd.Series(df.index + pd.Timedelta(seconds=TF_SECONDS[tf]), index=df.index, name="t")
    m = pd.merge_asof(close_time.to_frame(), ht.rename("ht").to_frame(), left_on="t", right_index=True, direction="backward")
    htr = pd.Series(m["ht"].fillna(0).values, index=df.index)

    # ---------- نظام الاتجاه ----------
    d = np.sign(e21 - e50)
    d = d.where(d == htr, 0)  # يجب أن يتوافق الفريمان
    stack = ((e9 > e21) & (e21 > e50)) | ((e9 < e21) & (e21 < e50))
    s_trend = (
        25 * (d != 0)
        + 20 * (stack & (np.sign(e9 - e21) == d))
        + 15 * ((np.sign(mh) == d) & ((mh - mh.shift(1)) * d > 0))
        + 25 * (((d > 0) & r.between(38, 56) & (r > r.shift(1))) | ((d < 0) & r.between(44, 62) & (r < r.shift(1))))
        + 15 * ((np.sign(body) == d) & (body.abs() > 0.5))
    ) * d
    s_trend = s_trend.where(ax >= 20, 0)

    # ---------- نظام التذبذب ----------
    buy = 30 * (pb <= 0.05) + 30 * (r < 30) + 25 * ((c > o) & (lower_wick > 0.35)) + 15 * (htr >= 0)
    sell = 30 * (pb >= 0.95) + 30 * (r > 70) + 25 * ((c < o) & (upper_wick > 0.35)) + 15 * (htr <= 0)
    s_range = pd.Series(np.where(buy > sell, buy, -sell), index=df.index).where(ax < 20, 0)
    s_range = s_range.where((pb <= 0.05) | (pb >= 0.95) | (r < 30) | (r > 70), 0)

    score = (s_trend + s_range).where((vol_ratio >= 0.6) & (vol_ratio <= 3.0), 0)
    out["score"] = score.fillna(0)
    out["close"], out["rsi"], out["adx"], out["pctb"] = c, r, ax, pb
    out["htf"], out["atr"], out["vol_ratio"] = htr, a, vol_ratio
    out["regime"] = np.where(ax >= 20, "اتجاه", "تذبذب")
    return out


def explain(row: pd.Series) -> list[str]:
    s = []
    s.append(f"النظام: {row['regime']} | ADX={row['adx']:.0f}")
    s.append(f"RSI={row['rsi']:.0f} | موقع بولينجر={row['pctb']:.2f}")
    s.append("اتجاه الفريم الأكبر: " + {1: "صاعد", -1: "هابط", 0: "محايد"}[int(row["htf"])])
    s.append(f"نسبة التذبذب ATR={row['vol_ratio']:.2f}")
    return s
