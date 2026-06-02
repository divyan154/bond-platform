"""Central configuration for the AI-Powered Bond Intelligence Platform."""
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import streamlit as st
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT_DIR / "raw_data"
OUTPUT_DIR = ROOT_DIR / "output"
CHROMA_DIR = OUTPUT_DIR / "chroma"
PROCESSED_DIR = OUTPUT_DIR / "processed"

for d in (RAW_DATA_DIR, OUTPUT_DIR, CHROMA_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)

UNIFIED_CSV = OUTPUT_DIR / "unified_bond_master.csv"
SQLITE_DB = OUTPUT_DIR / "bond_platform.db"
KG_FILE = OUTPUT_DIR / "knowledge_graph.gpickle"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_COLLECTION = "bond_corpus"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
API_BASE = f"http://{FLASK_HOST}:{FLASK_PORT}"

RESTRICTED_PHRASES = [
    "guaranteed return", "risk free profit", "sure shot",
    "100% safe", "no risk", "guaranteed profit", "assured return",
]
COMPLIANCE_DISCLAIMER = (
    "DISCLAIMER: All bond analytics shown are computed from verified data sources. "
    "This is not investment advice. Bond investments are subject to credit, interest "
    "rate and liquidity risks. Consult a SEBI-registered advisor before investing."
)
MIN_CONFIDENCE = 0.35
