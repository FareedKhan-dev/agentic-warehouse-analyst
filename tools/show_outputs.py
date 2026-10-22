"""Print the text outputs of an executed notebook, cell by cell, so runs can be reviewed in a terminal.

Usage: .venv/Scripts/python.exe tools/show_outputs.py notebook/runs/smoke.ipynb [--from N] [--to M] [--max 60]
"""
import argparse, json, sys

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--from", dest="start", type=int, default=0)
ap.add_argument("--to", dest="stop", type=int, default=10**9)
ap.add_argument("--max", type=int, default=60, help="max output lines per cell")
args = ap.parse_args()

nb = json.load(open(args.path, encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    if not (args.start <= i <= args.stop) or c["cell_type"] != "code":
        continue
    src = "".join(c["source"]).strip().splitlines()
    head = next((l for l in src if l.startswith("# ----")), src[0] if src else "")
    lines, images = [], 0
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            lines += "".join(o["text"]).rstrip("\n").splitlines()
        elif o.get("output_type") in ("execute_result", "display_data"):
            if "image/png" in o.get("data", {}):
                images += 1
            elif "text/plain" in o.get("data", {}):
                lines += "".join(o["data"]["text/plain"]).splitlines()
        elif o.get("output_type") == "error":
            lines += [f"ERROR {o['ename']}: {o['evalue']}"]
    print(f"\n=== [{i}] {head[:100]}" + (f"   (+{images} figure)" if images else ""))
    for l in lines[: args.max]:
        print("   " + l)
    if len(lines) > args.max:
        print(f"   ... {len(lines) - args.max} more lines")
