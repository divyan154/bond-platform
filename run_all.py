"""
One-click orchestrator.

Steps:
    1. (optional) zintellix data collection
    2. CSV consolidation -> output/unified_bond_master.csv
    3. SQLite + ChromaDB + Knowledge Graph build
    4. Launch Flask API + Streamlit UI in subprocesses

Usage:
    python run_all.py                  # full pipeline then launch services
    python run_all.py --no-live        # skip zintellix network ingest
    python run_all.py --build-only     # build databases, do not launch
    python run_all.py --serve-only     # skip rebuild, just launch
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))


def step_pipeline(try_live: bool):
    from data_pipeline.data_consolidator import run as run_pipe
    run_pipe(try_live=try_live)


def step_build_db():
    from database.build_all import build_all
    build_all()


def serve():
    py = sys.executable
    print("[run_all] starting Flask API ...")
    api = subprocess.Popen([py, "-m", "api.app"], cwd=ROOT)
    time.sleep(2)
    print("[run_all] starting Streamlit UI ...")
    ui = subprocess.Popen(
        [py, "-m", "streamlit", "run", "ui/streamlit_app.py",
         "--server.headless", "true"],
        cwd=ROOT,
    )
    try:
        api.wait()
        ui.wait()
    except KeyboardInterrupt:
        print("\n[run_all] shutting down ...")
        api.terminate(); ui.terminate()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-live",   action="store_true",
                   help="skip live zintellix ingest")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--serve-only", action="store_true")
    args = p.parse_args()

    if not args.serve_only:
        step_pipeline(try_live=not args.no_live)
        step_build_db()

    if args.build_only:
        return
    serve()


if __name__ == "__main__":
    main()
