"""اختبار تاريخي صادق: يعرض نسبة النجاح الحقيقية، مقارنةً بنسبة التعادل، على عيّنة تدريب وعيّنة خارجية."""
import math
import numpy as np
import pandas as pd


def wilson_low(w, n, z=1.96):
    if n == 0:
        return 0.0
    p = w / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def simulate(sc: pd.DataFrame, threshold: float, bars: int):
    close = sc["close"].values
    score = sc["score"].values
    res, i = [], 0
    while i < len(sc) - bars:
        if abs(score[i]) >= threshold:
            win = (close[i + bars] > close[i]) if score[i] > 0 else (close[i + bars] < close[i])
            res.append(bool(win))
            i += bars  # لا صفقة جديدة قبل انتهاء الحالية
        else:
            i += 1
    return res


def report(sc: pd.DataFrame, payout=0.85, thresholds=(60, 70, 80), bars_list=(1, 2, 3), split=0.6) -> pd.DataFrame:
    sc = sc.iloc[120:]  # فترة إحماء للمؤشرات
    cut = int(len(sc) * split)
    rows = []
    for th in thresholds:
        for b in bars_list:
            for name, part in (("train", sc.iloc[:cut]), ("OUT-of-sample", sc.iloc[cut:])):
                r = simulate(part, th, b)
                n, w = len(r), sum(r)
                wr = w / n if n else 0
                rows.append({"threshold": th, "expiry_bars": b, "sample": name, "trades": n,
                             "win_rate_%": round(100 * wr, 1),
                             "reliable_low_%": round(100 * wilson_low(w, n), 1),
                             "profit_per_trade_%": round(100 * (wr * payout - (1 - wr)), 1)})
    return pd.DataFrame(rows)
