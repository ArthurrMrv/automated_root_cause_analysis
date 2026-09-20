"""Run several RCAEval methods on one dataset and print an Avg@5 table.

    python compare.py --dataset re2-ob --method baro,prism,mars,tgfi,lstr,dterwr
"""
import argparse
import re
import subprocess
import sys
from os.path import abspath, dirname, join

FAULTS = ("CPU", "MEM", "DISK", "SOCKET", "DELAY", "LOSS")
ROOT = dirname(abspath(__file__))


def parse_scores(text):
    scores = {k: float(v) for k, v in re.findall(r"Avg@5-(\w+):\s+([0-9.]+)", text)}
    speed = re.search(r"Avg speed:\s+([0-9.]+)", text)
    return scores, float(speed.group(1)) if speed else None


def run(method, dataset, extra):
    out = join("output", method)
    cmd = [
        sys.executable, join(ROOT, "main.py"),
        "--method", method, "--dataset", dataset, "--output", out,
        *extra,
    ]
    print(f"\n=== {method} ===", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
    buf = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        buf.append(line)
    proc.wait()
    return proc.returncode, "".join(buf)


def print_table(dataset, rows):
    name_w = max(6, *(len(r[0]) for r in rows))
    cols = f"{{:<{name_w}}}  " + "  ".join(["{:>6}"] * 8)
    print(f"\n--- Comparison ({dataset}) ---")
    print(cols.format("method", *FAULTS, "avg", "speed"))
    for name, scores, speed, ok in rows:
        vals = [scores.get(f) for f in FAULTS]
        present = [v for v in vals if v is not None]
        avg = sum(present) / len(present) if present else None
        cells = [name] + [f"{v:.2f}" if v is not None else "-" for v in vals + [avg, speed]]
        line = cols.format(*cells)
        print(line if ok else f"{line}  FAIL")


def main():
    parser = argparse.ArgumentParser(description="Compare RCAEval methods")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True, help="Comma-separated methods")
    args, extra = parser.parse_known_args()

    rows = []
    for method in [m.strip() for m in args.method.split(",") if m.strip()]:
        code, text = run(method, args.dataset, extra)
        scores, speed = parse_scores(text)
        rows.append((method, scores, speed, code == 0))
    print_table(args.dataset, rows)


if __name__ == "__main__":
    main()
