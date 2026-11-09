#!/usr/bin/env python3
"""
محلل إشارات العملات الرقمية لمنصة IQ Option (تنفيذ يدوي منك).
  python main.py backtest --pair BTC/USD --tf 5m
  python main.py scan --pairs BTC/USD,ETH/USD --tf 5m --threshold 70
  python main.py backtest --demo        # تجربة بدون إنترنت
"""
import argparse
import sys
import time
from datetime import timedelta
from zoneinfo import ZoneInfo

import pandas as pd

import data
import strategy
from backtest import report

HTF = {"1m": "15m", "5m": "1h", "15m": "1h", "30m": "1h", "1h": "1h"}
DEFAULT_PAIRS = "BTC/USD,ETH/USD,XRP/USD,LTC/USD,SOL/USD,ADA/USD,DOGE/USD,BNB/USD"
G, R, Y, B, X = "\033[92m", "\033[91m", "\033[93m", "\033[1m", "\033[0m"


def load(pair, tf, limit, demo=False):
    if demo:
        base = data.synthetic(limit, "1m" if tf == "1m" else "5m", seed=abs(hash(pair)) % 1000)
        base = base if tf in ("1m", "5m") else data.resample(base, tf)
        return base, data.resample(base, HTF[tf])
    df = data.fetch(pair, tf, limit)
    need = max(200, int(limit * data.TF_SECONDS[tf] / data.TF_SECONDS[HTF[tf]]) + 120)
    return df, data.fetch(pair, HTF[tf], min(need, 3000))


def cmd_backtest(a):
    pairs = [p.strip() for p in a.pairs.split(",")] if a.pairs else [a.pair]
    for p in pairs:
        df, htf = load(p, a.tf, a.bars, a.demo)
        sc = strategy.compute(df, htf, a.tf, HTF[a.tf])
        print(f"\n{B}=== {p} | فريم {a.tf} | {len(df)} شمعة{' (بيانات تجريبية)' if a.demo else ''} ==={X}")
        print(report(sc, a.payout).to_string(index=False))
    be = 100 / (1 + a.payout)
    print(f"\nنسبة التعادل عند عائد {a.payout*100:.0f}% = {be:.1f}% . لا تتداول إلا إذا كان 'أدنى حد موثوق' "
          f"فوقها في عيّنة 'خارج العيّنة' وليس التدريب فقط.")


def cmd_research(a):
    import research
    pairs = [p.strip() for p in a.pairs.split(",")]
    nb = {"5m": 20000, "15m": 10000}
    allrows, passed = [], 0
    for tf in [t.strip() for t in a.tfs.split(",")]:
        n = a.bars or (3000 if a.demo else nb.get(tf, 8000))
        print(f"\n=== timeframe {tf} | up to {n} candles per pair ===")
        res = research.run(pairs, tf, HTF[tf], n, lambda p, t, l: load(p, t, l, a.demo), a.payout)
        if res is None:
            continue
        top, full, be = res
        allrows.append(full)
        print(f"\nBreak-even win rate = {be:.1f}%.  TEST columns are on data never used for selection.")
        print(top.to_string(index=False) if len(top) else "no config had enough trades")
        passed += int((top.verdict == "PASS").sum())
    if allrows:
        pd.concat(allrows).to_csv("research_results.csv", index=False)
    print("\nRESULT:", f"{passed} config(s) PASSED the final test." if passed else
          "NO config passed the final test. Do NOT trade real money with this.")


def cmd_scan(a):
    tz = ZoneInfo(a.tz)
    pairs = [p.strip() for p in a.pairs.split(",")]
    period = data.TF_SECONDS[a.tf]
    pending, stats = [], {"w": 0, "l": 0}
    print(f"{B}بدأ المسح: {len(pairs)} أزواج | فريم {a.tf} | عتبة {a.threshold} | الوقت المحلي {a.tz}{X}  (Ctrl+C للإيقاف)")
    while True:
        now = time.time()
        nxt = (int(now // period) + 1) * period
        time.sleep(max(0, nxt - now) + a.delay)
        stamp = pd.Timestamp.now(tz=tz).strftime("%H:%M:%S")
        print(f"\n[{stamp}] إغلاق شمعة جديدة")
        found = False
        for p in pairs:
            try:
                df, htf = load(p, a.tf, 400, a.demo)
                sc = strategy.compute(df, htf, a.tf, HTF[a.tf])
            except Exception as e:
                print(f"  {p}: خطأ بيانات ({e})")
                continue
            row, last_t = sc.iloc[-1], sc.index[-1]
            # تقييم الصفقات السابقة
            for s in list(pending):
                if s["pair"] == p and last_t >= s["exit_bar"]:
                    exit_close = sc.loc[s["exit_bar"], "close"] if s["exit_bar"] in sc.index else row["close"]
                    won = (exit_close > s["price"]) if s["dir"] == "CALL" else (exit_close < s["price"])
                    stats["w" if won else "l"] += 1
                    pending.remove(s)
                    n = stats["w"] + stats["l"]
                    print(f"  {'✔ ربح' if won else '✘ خسارة'} {p} {s['dir']} | الحصيلة {stats['w']}/{n} = {100*stats['w']/n:.0f}%")
            if abs(row["score"]) >= a.threshold:
                found = True
                d = "CALL" if row["score"] > 0 else "PUT"
                col, ar = (G, "شراء ▲ (Call/Higher)") if d == "CALL" else (R, "بيع ▼ (Put/Lower)")
                entry = (last_t + timedelta(seconds=period)).tz_convert(tz)
                exp = entry + timedelta(seconds=period * a.expiry_bars)
                print(f"{col}{B}  ★ {p}: {ar} | القوة {abs(row['score']):.0f}/100{X}")
                print(f"    ادخل عند: {entry:%H:%M:%S} (بداية الشمعة القادمة) | الانتهاء: {exp:%H:%M} ({a.expiry_bars*period//60} دقيقة)")
                print(f"    السعر الآن: {row['close']:.5g} | " + " | ".join(strategy.explain(row)))
                pending.append({"pair": p, "dir": d, "price": row["close"],
                                "exit_bar": last_t + timedelta(seconds=period * a.expiry_bars)})
                print("\a", end="")
        if not found:
            print(f"  {Y}لا توجد فرصة قوية — الانتظار أفضل من الدخول الضعيف.{X}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("backtest", "scan", "research"):
        s = sub.add_parser(name)
        s.add_argument("--tf", default="5m", choices=list(HTF))
        s.add_argument("--demo", action="store_true", help="بيانات اصطناعية بدون إنترنت")
        s.add_argument("--tz", default="Asia/Riyadh")
        if name == "backtest":
            s.add_argument("--pair", default="BTC/USD")
            s.add_argument("--pairs", default="")
            s.add_argument("--bars", type=int, default=5000)
            s.add_argument("--payout", type=float, default=0.85, help="عائد المنصة، مثلاً 0.85")
        elif name == "research":
            s.add_argument("--pairs", default=DEFAULT_PAIRS)
            s.add_argument("--tfs", default="5m,15m")
            s.add_argument("--bars", type=int, default=0)
            s.add_argument("--payout", type=float, default=0.85)
        else:
            s.add_argument("--pairs", default=DEFAULT_PAIRS)
            s.add_argument("--threshold", type=float, default=70)
            s.add_argument("--expiry-bars", type=int, default=1, dest="expiry_bars")
            s.add_argument("--delay", type=float, default=2.0, help="ثوانٍ بعد الإغلاق قبل الجلب")
    a = ap.parse_args()
    try:
        {"backtest": cmd_backtest, "scan": cmd_scan, "research": cmd_research}[a.cmd](a)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
