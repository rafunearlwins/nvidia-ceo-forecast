"""Quantile forecaster for Polymarket "Next CEO of NVIDIA" Yes prices.

For each market: empirical distribution of h-minute price changes over the
market's recent history (skipping its noisy launch period), shrunk toward
zero, centred on the last price, interpolated in log-horizon.
"""
import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

ANCHORS = [1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 100, 140, 180, 240]
WINDOW = 2880
SKIP = 360
SHRINK = 0.6


def quantile(sorted_vals, t):
    k = t * (len(sorted_vals) - 1)
    i = int(k)
    f = k - i
    if i + 1 >= len(sorted_vals):
        return sorted_vals[i]
    return sorted_vals[i] * (1 - f) + sorted_vals[i + 1] * f


def change_quantiles(hist, h, levels):
    if len(hist) <= h + 30:
        return None
    d = sorted(hist[t + h] - hist[t] for t in range(len(hist) - h))
    return [quantile(d, t) for t in levels]


def market_rows(prices, horizon, levels):
    x = prices[-1]
    if len(prices) > SKIP + 600:
        hist = prices[SKIP:]
    else:
        hist = prices[min(len(prices) // 3, SKIP):]
    hist = hist[-WINDOW:]
    res = {h: change_quantiles(hist, h, levels) for h in ANCHORS}
    avail = [h for h in ANCHORS if res[h] is not None]
    rows = []
    for h in range(1, horizon + 1):
        if not avail:
            dq = [0.0] * len(levels)
        else:
            lo = max([a for a in avail if a <= h], default=avail[0])
            hi = min([a for a in avail if a >= h], default=avail[-1])
            if lo == hi:
                dq = res[lo]
            else:
                w = (math.log(h) - math.log(lo)) / (math.log(hi) - math.log(lo))
                dq = [a * (1 - w) + b * w for a, b in zip(res[lo], res[hi])]
        r = [min(1.0, max(0.0, x + SHRINK * v)) for v in dq]
        for j in range(1, len(r)):
            r[j] = max(r[j], r[j - 1])
        rows.append([round(v, 6) + 0.0 for v in r])
    return rows


def forecast_case(case):
    horizon = case["horizon_minutes"]
    levels = case["quantiles"]
    return {
        "forecasts": [
            {"id": m["id"], "q": market_rows(m["prices"], horizon, levels)}
            for m in case["markets"]
        ]
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name in sorted(os.listdir(INPUT_DIR)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(INPUT_DIR, name)) as f:
            case = json.load(f)
        out = forecast_case(case)
        with open(os.path.join(OUTPUT_DIR, name[:-3] + ".out"), "w") as f:
            json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
