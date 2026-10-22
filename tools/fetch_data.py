"""Download the prebuilt TPC-DS DuckDB files to D:\\tpcds_agent with resume and size checks.

Usage: .venv/Scripts/python.exe tools/fetch_data.py
Safe to re-run: finished files are verified and skipped, partial files resume.
"""
import sys, time
from pathlib import Path

import httpx

sys.stdout.reconfigure(encoding="utf-8")
DATA = Path("D:/tpcds_agent")
FILES = [  # (url, local name, exact bytes)
    ("https://blobs.duckdb.org/data/tpcds-sf100.db", "tpcds_sf100.duckdb", 28_609_622_016),
]
CHUNK = 8 * 1024 * 1024


def fetch(url, name, expected):
    final, part = DATA / name, DATA / (name + ".part")
    if final.exists():
        size = final.stat().st_size
        status = "OK" if size == expected else "SIZE MISMATCH"
        print(f"[fetch] {name}: already present, {size:,} bytes, {status}", flush=True)
        return size == expected
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    print(f"[fetch] {name}: starting at {have:,} of {expected:,} bytes", flush=True)
    t0, done, next_mark = time.time(), have, 0
    with httpx.stream("GET", url, headers=headers, timeout=60, follow_redirects=True) as r:
        if have and r.status_code != 206:
            raise RuntimeError(f"server ignored the range request (status {r.status_code})")
        r.raise_for_status()
        with open(part, "ab") as f:
            for chunk in r.iter_bytes(CHUNK):
                f.write(chunk)
                done += len(chunk)
                pct = 100 * done / expected
                if pct >= next_mark:
                    rate = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                    print(f"[fetch] {name}: {pct:5.1f}%  {done/1e9:6.2f} GB  {rate:6.1f} MB/s", flush=True)
                    next_mark = int(pct) + 2
    size = part.stat().st_size
    if size != expected:
        print(f"[fetch] {name}: got {size:,}, expected {expected:,}; re-run to resume", flush=True)
        return False
    part.rename(final)
    mins = (time.time() - t0) / 60
    print(f"[fetch] {name}: DONE, {size:,} bytes verified, {mins:.1f} min", flush=True)
    return True


def fetch_with_resume(url, name, expected, max_restarts=60):
    """Servers drop long downloads; resume from the partial file until it is complete."""
    for attempt in range(1, max_restarts + 1):
        try:
            if fetch(url, name, expected):
                return True
        except (httpx.HTTPError, OSError) as e:
            print(f"[fetch] {name}: connection lost ({type(e).__name__}), resuming (restart {attempt})", flush=True)
            time.sleep(5)
    return False


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    ok = all(fetch_with_resume(*f) for f in FILES)
    print("[fetch] all files verified" if ok else "[fetch] some files incomplete", flush=True)
