"""Check an executed notebook against the writing rules.

Usage: .venv/Scripts/python.exe tools/lint_notebook.py notebook/runs/sf100_full.ipynb [--numbers]

Checks
  1. no em dash (U+2014) or en dash (U+2013) anywhere: markdown, code or outputs
  2. no stock phrases that make text read as machine-written
  3. no leftover TODO_INTERPRET placeholders
  4. (--numbers) every number written in a markdown cell appears in the output of an earlier
     code cell, so prose can only quote what the notebook actually printed
"""
import argparse, json, re, sys

sys.stdout.reconfigure(encoding="utf-8")
PHRASES = ["delve", "tapestry", "testament to", "in today's", "it's worth noting", "it is worth noting",
           "furthermore", "moreover", "paramount", "unlock", "empower", "game-changer", "game changer",
           "cutting-edge", "harness the power", "deep dive", "dive into", "realm", "navigate the", "embark",
           "in conclusion", "additionally,", "seamless", "plethora", "myriad", "elevate", "supercharge",
           "revolutionize", "revolutionise", "unleash", "at the end of the day", "needless to say"]
NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")


def text_of(cell) -> str:
    return "".join(cell.get("source", ""))


def outputs_of(cell) -> str:
    out = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            out.append("".join(o.get("text", "")))
        elif "data" in o and "text/plain" in o["data"]:
            out.append("".join(o["data"]["text/plain"]))
    return "\n".join(out)


def norm_number(s: str) -> str:
    return s.replace(",", "").rstrip("%")


def backed(v: str, pool: set[float], percent: bool = False) -> bool:
    """A prose number is backed if some printed number, rounded to the prose's precision, equals it.
    A percentage may also come from a printed fraction (0.961 in a table, 96.1% in prose)."""
    d = len(v.split(".")[1]) if "." in v else 0
    x = float(v)
    return any(abs(round(p, d) - x) < 1e-9 or (percent and abs(round(100 * p, d) - x) < 1e-9) for p in pool)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--numbers", action="store_true")
    args = ap.parse_args()
    nb = json.load(open(args.path, encoding="utf-8"))
    problems = 0
    seen_output = ""
    for i, c in enumerate(nb["cells"]):
        src = text_of(c)
        blobs = [("source", src)] + ([("output", outputs_of(c))] if c["cell_type"] == "code" else [])
        for where, blob in blobs:
            for ch, name in (("—", "em dash"), ("–", "en dash")):
                if ch in blob:
                    problems += 1
                    line = next(l for l in blob.splitlines() if ch in l)
                    print(f"[{i}] {name} in {where}: {line.strip()[:110]}")
            low = blob.lower()
            for p in PHRASES:
                if re.search(r"\b" + re.escape(p), low):
                    problems += 1
                    print(f"[{i}] phrase '{p}' in {where}")
        if "TODO_INTERPRET" in src:
            problems += 1
            print(f"[{i}] leftover TODO_INTERPRET")
        if c["cell_type"] == "code":
            seen_output += "\n" + outputs_of(c)
        elif args.numbers and c["cell_type"] == "markdown" and "toc" not in c.get("metadata", {}).get("tags", []):
            body = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
            pool = {float(norm_number(n)) for n in NUM.findall(seen_output)}
            for n in NUM.findall(body):
                v = norm_number(n)
                if len(v.replace("-", "").replace(".", "")) <= 1:   # single digits are rarely claims
                    continue
                if not backed(v, pool, percent=n.endswith("%")):
                    problems += 1
                    print(f"[{i}] number {n} in markdown not found in any earlier output")
    print(f"\n{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
