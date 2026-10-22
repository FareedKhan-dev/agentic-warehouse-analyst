"""Build the .ipynb from the jupytext source and execute it headless with papermill.

Usage:
  .venv/Scripts/python.exe tools/run_nb.py --smoke          # scale 1 copy, checks code paths only
  .venv/Scripts/python.exe tools/run_nb.py --name full      # the real SF100 run
  .venv/Scripts/python.exe tools/run_nb.py -p KEY VALUE ... # extra papermill parameters

Every Jupyter/IPython/matplotlib cache is pointed inside the project folder.
"""
import argparse, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SRC = ROOT / "notebook" / "agentic_data_analyst.py"
NB = ROOT / "notebook" / "agentic_data_analyst.ipynb"
RUNS = ROOT / "notebook" / "runs"


def project_env() -> dict:
    cache = ROOT / ".cache"
    env = dict(os.environ)
    env.update(
        JUPYTER_CONFIG_DIR=str(cache / "jupyter" / "config"),
        JUPYTER_RUNTIME_DIR=str(cache / "jupyter" / "runtime"),
        JUPYTER_DATA_DIR=str(ROOT / ".venv" / "share" / "jupyter"),
        IPYTHONDIR=str(cache / "ipython"),
        MPLCONFIGDIR=str(cache / "matplotlib"),
        PYTHONIOENCODING="utf-8",
    )
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("-p", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"))
    args = ap.parse_args()

    env = project_env()
    RUNS.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(PY), "-m", "jupytext", "--to", "ipynb", str(SRC), "-o", str(NB)],
                   check=True, env=env)
    name = args.name or ("smoke" if args.smoke else "run")
    out = RUNS / f"{name}.ipynb"
    cmd = [str(PY), "-m", "papermill", str(NB), str(out), "-k", "tpcds-agent",
           "--cwd", str(NB.parent)]
    if args.smoke:
        cmd += ["-p", "SMOKE_TEST", "True"]
    for k, v in args.p:
        cmd += ["-p", k, v]
    print("[run]", " ".join(cmd[2:]), flush=True)
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
