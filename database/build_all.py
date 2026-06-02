"""One-shot builder for SQLite + ChromaDB + Knowledge Graph from unified CSV."""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import UNIFIED_CSV
from database.sqlite_store import build_sqlite
from database.vector_store import build_vector_store
from database.knowledge_graph import build_graph


def build_all():
    if not UNIFIED_CSV.exists():
        raise FileNotFoundError(
            f"{UNIFIED_CSV} missing. Run data_pipeline.data_consolidator first."
        )
    df = pd.read_csv(UNIFIED_CSV)
    print("\n[BUILD] SQLite ...");        build_sqlite(df)
    print("\n[BUILD] Knowledge Graph ..."); build_graph(df)
    print("\n[BUILD] Vector Store ...");   build_vector_store(df)
    print("\n[BUILD] All databases ready.")


if __name__ == "__main__":
    build_all()
