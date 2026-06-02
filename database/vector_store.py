"""
ChromaDB vector store for semantic bond search.

Each bond is embedded as a natural-language descriptor; ChromaDB stores
the vector + metadata so RAG queries can retrieve grounded facts.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL, UNIFIED_CSV

_chroma_client = None
_collection = None
_embedder = None


def _get_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print(f"[vector] loading embedding model: {EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_collection(reset: bool = False):
    global _collection
    client = _get_client()
    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
        _collection = None
    if _collection is None:
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _bond_to_text(row: pd.Series) -> str:
    parts = [
        f"ISIN {row.get('ISIN','')}",
        f"Issuer {row.get('IssuerName','') or row.get('Symbol','')}",
        f"Symbol {row.get('Symbol','')}",
        f"Sector {row.get('Sector','') or 'Unspecified'}",
        f"Country {row.get('Country','') or 'India'}",
        f"Bond type {row.get('BondType','') or 'Corporate'}",
        f"Rating {row.get('Rating','') or 'Unrated'}",
        f"Coupon {row.get('CouponRate','') or 'N/A'}%",
        f"Maturity {row.get('MaturityDate','')}",
        f"Face value {row.get('FaceValue','')}",
        f"Last traded price {row.get('LastTradedPrice','')}",
        f"Yield {row.get('Yield','')}",
        f"Duration {row.get('Duration','')}",
        f"Spread {row.get('Spread','')}",
        f"Currency {row.get('Currency','') or 'INR'}",
    ]
    return " | ".join(str(p) for p in parts if str(p).split(maxsplit=1)[-1] not in ("", "nan", "None"))


def _clean_meta(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if pd.isna(v) if not isinstance(v, str) else v in ("", "nan", "None"):
            out[k] = ""
        elif isinstance(v, (int, float, bool, str)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def build_vector_store(df: pd.DataFrame | None = None,
                       batch_size: int = 64) -> int:
    if df is None:
        df = pd.read_csv(UNIFIED_CSV)
    coll = _get_collection(reset=True)
    embedder = _get_embedder()

    ids, texts, metas = [], [], []
    for _, row in df.iterrows():
        ids.append(str(row["ISIN"]))
        texts.append(_bond_to_text(row))
        metas.append(_clean_meta({
            "ISIN":          row.get("ISIN", ""),
            "Symbol":        row.get("Symbol", ""),
            "IssuerName":    row.get("IssuerName", ""),
            "Sector":        row.get("Sector", ""),
            "Rating":        row.get("Rating", ""),
            "BondType":      row.get("BondType", ""),
            "Country":       row.get("Country", ""),
            "CouponRate":    row.get("CouponRate", ""),
            "MaturityDate":  str(row.get("MaturityDate", "")),
            "LastTradedPrice": row.get("LastTradedPrice", ""),
            "Yield":         row.get("Yield", ""),
            "Source":        row.get("Source", ""),
        }))

    total = 0
    for i in range(0, len(ids), batch_size):
        chunk_ids = ids[i:i + batch_size]
        chunk_txt = texts[i:i + batch_size]
        chunk_meta = metas[i:i + batch_size]
        vecs = embedder.encode(chunk_txt, show_progress_bar=False).tolist()
        coll.add(ids=chunk_ids, embeddings=vecs,
                 documents=chunk_txt, metadatas=chunk_meta)
        total += len(chunk_ids)
    print(f"[vector] embedded {total} bonds -> {CHROMA_DIR.name}/")
    return total


def semantic_search(query: str, k: int = 5, where: dict | None = None) -> list[dict]:
    coll = _get_collection()
    embedder = _get_embedder()
    vec = embedder.encode([query]).tolist()
    res = coll.query(query_embeddings=vec, n_results=k, where=where)
    hits = []
    if not res.get("ids"):
        return hits
    for i, _id in enumerate(res["ids"][0]):
        hits.append({
            "id":       _id,
            "document": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i] if res.get("distances") else None,
            "score":    1 - res["distances"][0][i] if res.get("distances") else None,
        })
    return hits


if __name__ == "__main__":
    build_vector_store()
