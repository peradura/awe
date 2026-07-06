#!/usr/bin/env python3
"""Aggregate multi-seed sweep logs into mean +/- std tables.

Usage: python3 scripts/aggregate.py results
Parses results/<exp>_s<seed>.log for lines like
  `  both    | acc  25.7% | steps 5.38 | ...`   (4-way ablations)
  `  ent      2.8181  46.9%   5.91 ...`          (bake-off arms)
and any `corr... = +0.123` line.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ABL = re.compile(r"^\s+(\S+)\s+\| acc\s+([\d.]+)% \| steps\s+([\d.]+)")
BAK = re.compile(r"^\s+(ent|recon|rnorm|dstate|dent)\s+(-?[\d.]+)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)%\s+([+-][\d.]+)")
COR = re.compile(r"corr[^=]*=\s*([+-]?\d+\.\d+)")


def fmt(xs):
    if len(xs) == 1:
        return f"{xs[0]:.2f} (n=1)"
    return f"{mean(xs):.2f} ± {stdev(xs):.2f} (n={len(xs)})"


def main(root):
    acc = defaultdict(lambda: defaultdict(list))    # exp -> key -> values
    steps = defaultdict(lambda: defaultdict(list))
    corr = defaultdict(lambda: defaultdict(list))
    for log in sorted(Path(root).glob("*_s*.log")):
        exp = re.sub(r"_s\d+$", "", log.stem)
        for line in log.read_text().splitlines():
            m = ABL.match(line)
            if m:
                acc[exp][m.group(1)].append(float(m.group(2)))
                steps[exp][m.group(1)].append(float(m.group(3)))
                continue
            m = BAK.match(line)
            if m:
                acc[exp][m.group(1)].append(float(m.group(3)))
                steps[exp][m.group(1)].append(float(m.group(4)))
                corr[exp][m.group(1)].append(float(m.group(6)))
                continue
            m = COR.search(line)
            if m:
                corr[exp]["(headline)"].append(float(m.group(1)))
    if not acc and not corr:
        print(f"no *_s<seed>.log files found under {root}")
        return
    for exp in sorted(set(acc) | set(corr)):
        print(f"\n== {exp} ==")
        for k in acc.get(exp, {}):
            line = f"  {k:9s} acc {fmt(acc[exp][k]):24s}"
            if steps[exp].get(k):
                line += f" steps {fmt(steps[exp][k])}"
            if corr[exp].get(k):
                line += f" corr {fmt(corr[exp][k])}"
            print(line)
        if corr[exp].get("(headline)") and exp not in ("bakeoff_rule", "bakeoff_reachp"):
            print(f"  {'corr':9s} {fmt(corr[exp]['(headline)'])}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
