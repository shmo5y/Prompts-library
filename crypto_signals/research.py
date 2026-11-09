"""
بحث صادق عن إعدادات تربح فعلاً.
البيانات تُقسم زمنياً: 50% تدريب | 25% تحقق | 25% اختبار نهائي لا يُستخدم في الاختيار أبداً.
نختار الإعدادات بالتدريب+التحقق فقط، ثم نعرض أداءها على الاختبار النهائي (الرقم الصادق).
"""
import numpy as np
import pandas as pd
import strategy
from backtest import wilson_low

MODES = {"trend": "score_trend", "range": "score_range", "both": "score"}
THRESHOLDS = (50, 60, 70, 80, 90)
BARS = (1, 2, 3, 4)


def _wins(close, score, thr, bars):
    w = n = nxt = 0
    for i in np.flatnonzero(np.abs(score) >= thr):
        if i < nxt or i + bars >= len(close):
            continue
        n += 1
        w += bool(close[i + bars] > close[i]) if score[i] > 0 else bool(close[i + bars] < close[i])
        nxt = i + bars
    return w, n


def run(pairs, tf, htf_tf, nbars, loader, payout):
    be = 100 / (1 + payout)
    series = []
    for p in pairs:
        try:
            df, htf = loader(p, tf, nbars)
            sc = strategy.compute(df, htf, tf, htf_tf).iloc[120:]
            series.append(sc)
            print(f"  loaded {p}: {len(sc)} candles")
        except Exception as e:
            print(f"  skip {p}: {e}")
    if not series:
        return None
    rows = []
    for mode, col in MODES.items():
        for thr in THRESHOLDS:
            for b in BARS:
                tot = {k: [0, 0] for k in ("train", "val", "test")}
                for sc in series:
                    n = len(sc)
                    cuts = {"train": (0, int(n * .5)), "val": (int(n * .5), int(n * .75)), "test": (int(n * .75), n)}
                    c, s = sc["close"].values, sc[col].values
                    for k, (lo, hi) in cuts.items():
                        w, m = _wins(c[lo:hi], s[lo:hi], thr, b)
                        tot[k][0] += w
                        tot[k][1] += m
                wr = {k: (100 * v[0] / v[1] if v[1] else 0) for k, v in tot.items()}
                rows.append({"tf": tf, "mode": mode, "thr": thr, "bars": b,
                             "n_train": tot["train"][1], "n_val": tot["val"][1], "n_test": tot["test"][1],
                             "win_train": round(wr["train"], 1), "win_val": round(wr["val"], 1),
                             "win_TEST": round(wr["test"], 1),
                             "low_TEST": round(100 * wilson_low(*tot["test"]), 1)})
    df = pd.DataFrame(rows)
    ok = df[(df.n_train >= 150) & (df.n_val >= 75) & (df.n_test >= 75)].copy()
    ok["pick"] = ok[["win_train", "win_val"]].min(axis=1)
    top = ok.sort_values("pick", ascending=False).head(8).copy()
    top["verdict"] = np.where(top.low_TEST > be, "PASS", np.where(top.win_TEST > be, "WEAK", "FAIL"))
    return top.drop(columns="pick"), df, be
