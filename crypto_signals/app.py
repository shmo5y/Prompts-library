#!/usr/bin/env python3
"""الواجهة الرسومية: شغّله بضغطتين، تُفتح صفحة في المتصفح تحلل الأزواج وتعطي (شراء / بيع / لا تنفذ)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)


def _bootstrap():
    """أول تشغيل: يجهّز المكتبات تلقائياً بدون أي أوامر منك."""
    try:
        import numpy, pandas, requests  # noqa: F401
        return
    except ImportError:
        pass
    if os.environ.get("CS_BOOT"):
        sys.exit("تعذّر تجهيز المكتبات، تأكد من الإنترنت ثم أعد التشغيل.")
    win = os.name == "nt"
    venv = os.path.join(HERE, "venv")
    py = os.path.join(venv, "Scripts" if win else "bin", "python.exe" if win else "python")
    if not os.path.exists(py):
        print("أول تشغيل: جارٍ تجهيز البرنامج، انتظر دقيقة...")
        subprocess.check_call([sys.executable, "-m", "venv", venv])
    subprocess.check_call([py, "-m", "pip", "install", "-q", "-r", os.path.join(HERE, "requirements.txt")])
    os.environ["CS_BOOT"] = "1"
    if win:
        sys.exit(subprocess.call([py, os.path.abspath(__file__)] + sys.argv[1:]))
    os.execv(py, [py, os.path.abspath(__file__)] + sys.argv[1:])


_bootstrap()

import json
import math
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

import data
import main
import strategy

DEMO = "--demo" in sys.argv
COL = {"trend": "score_trend", "range": "score_range", "both": "score"}
LOG = os.path.join(HERE, "signals_log.json")
LOCK = threading.Lock()
ALL_PAIRS = ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "SOL/USD", "ADA/USD", "DOGE/USD", "BNB/USD",
             "DOT/USD", "LINK/USD", "AVAX/USD", "TRX/USD", "BCH/USD", "XLM/USD", "ATOM/USD"]


def num(x, nd=2):
    try:
        x = float(x)
        return None if math.isnan(x) else round(x, nd)
    except Exception:
        return None


def read_log():
    try:
        with open(LOG, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def write_log(rows):
    with open(LOG, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)


def resolve_and_log(pair, tf, mode, expiry, sc, action, bar_ts, price):
    period = data.TF_SECONDS[tf]
    with LOCK:
        rows = read_log()
        for s in rows:
            if s["status"] != "pending" or s["pair"] != pair or s["tf"] != tf:
                continue
            exit_t = pd.Timestamp(s["bar_ts"] + s["expiry"] * period, unit="s", tz="UTC")
            if exit_t in sc.index:
                ex = float(sc.loc[exit_t, "close"])
                win = ex > s["price"] if s["dir"] == "CALL" else ex < s["price"]
                s["status"], s["exit_price"] = ("win" if win else "loss"), ex
            elif sc.index[0] > exit_t:
                s["status"] = "unknown"
        if action != "NONE" and not any(s["pair"] == pair and s["tf"] == tf and s["mode"] == mode and s["bar_ts"] == bar_ts for s in rows):
            rows.append({"pair": pair, "tf": tf, "mode": mode, "dir": action, "bar_ts": bar_ts,
                         "expiry": expiry, "price": price, "status": "pending"})
        write_log(rows)


def analyze_pair(pair, tf, mode, thr, expiry):
    try:
        df, htf = main.load(pair, tf, 400, DEMO)
        sc = strategy.compute(df, htf, tf, main.HTF[tf])
        row, t = sc.iloc[-1], sc.index[-1]
        score = float(row[COL[mode]])
        action = "CALL" if score >= thr else "PUT" if score <= -thr else "NONE"
        bar_ts = int(t.timestamp())
        resolve_and_log(pair, tf, mode, expiry, sc, action, bar_ts, float(row["close"]))
        period = data.TF_SECONDS[tf]
        return {"pair": pair, "action": action, "score": round(score), "price": num(row["close"], 6),
                "entry_ts": bar_ts + period, "expiry_ts": bar_ts + period * (1 + expiry),
                "minutes": period * expiry // 60, "period": period,
                "info": {"regime": row["regime"], "adx": num(row["adx"], 0), "rsi": num(row["rsi"], 0),
                         "pctb": num(row["pctb"]), "htf": int(row["htf"]), "vol": num(row["vol_ratio"])}}
    except Exception as e:
        return {"pair": pair, "action": "ERROR", "error": str(e)[:160]}


def stats():
    rows = read_log()
    w = sum(s["status"] == "win" for s in rows)
    l = sum(s["status"] == "loss" for s in rows)
    pend = sum(s["status"] == "pending" for s in rows)
    return {"wins": w, "losses": l, "pending": pend,
            "rate": round(100 * w / (w + l), 1) if w + l else None, "last": rows[-15:][::-1]}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json; charset=utf-8", code=200):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            self._send(open(os.path.join(HERE, "ui.html"), encoding="utf-8").read(), "text/html; charset=utf-8")
        elif u.path == "/api/pairs":
            self._send(json.dumps(ALL_PAIRS))
        elif u.path == "/api/stats":
            self._send(json.dumps(stats(), ensure_ascii=False))
        elif u.path == "/api/analyze":
            pairs = [p for p in q.get("pairs", "").split(",") if p]
            tf, mode = q.get("tf", "15m"), q.get("mode", "range")
            thr, ex = float(q.get("thr", 60)), int(q.get("expiry", 1))
            if tf not in main.HTF or mode not in COL:
                return self._send("{}", code=400)
            with ThreadPoolExecutor(6) as pool:
                res = list(pool.map(lambda p: analyze_pair(p, tf, mode, thr, ex), pairs))
            self._send(json.dumps({"results": res, "now": int(pd.Timestamp.now(tz="UTC").timestamp())}, ensure_ascii=False))
        else:
            self._send("{}", code=404)


def run():
    port = 8765
    while True:
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), H)
            break
        except OSError:
            port += 1
    url = f"http://127.0.0.1:{port}"
    print(f"\nالبرنامج يعمل على: {url}\nاترك هذه النافذة مفتوحة. للإغلاق اضغط Ctrl+C أو أغلق النافذة.")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
