# AI-Powered Bond Intelligence & Execution Platform

A Harvey-AI-style fixed-income workbench for the Indian bond market.
Implements the platform described in the project brief
(`AI-Powered Bond Intelligence & Execution Platform`):
unified Bond Master Database, Knowledge Graph, deterministic financial
analytics, retrieval-grounded AI Copilot, SEBI-aligned compliance
guardrails, Flask REST API, and an 8-page Streamlit dashboard.

---

## ⚡ Quick Start (5 commands)

After unzipping, open **PowerShell** in the `bond_platform` folder and run:

```powershell
# 1. (recommended) create a virtual env
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. install pinned dependencies
pip install -r requirements-freeze.txt

# 3. add an LLM key (optional — Copilot still works without it)
copy .env.example .env
# then open .env in any editor and paste your key:
#   GEMINI_API_KEY=AIza...        (free at https://aistudio.google.com/apikey)
# OR
#   OPENAI_API_KEY=sk-...

# 4. databases are pre-built and shipped in output/.
#    To rebuild from scratch from raw_data/, run:
python run_all.py --build-only --no-live

# 5. launch API (port 5000) + Streamlit UI (port 8501)
python run_all.py --serve-only
```

Then open **http://localhost:8501** in your browser.

> **Note on Python version:** the requirements were frozen against Python
> 3.14. Python 3.11+ will work too. If you're on 3.11/3.12 and `pip install`
> complains about a specific version, replace `requirements-freeze.txt`
> with `requirements.txt` (looser version pins) and it should resolve.

---

## 🗂️ What's in this zip

```
bond_platform/
├── README.md                       ← this file
├── requirements.txt                ← curated dependency list
├── requirements-freeze.txt         ← exact pinned versions
├── .env.example                    ← template for API keys
├── config.py                       ← central config (paths, env, disclaimer)
├── run_all.py                      ← one-click orchestrator
│
├── raw_data/                       ← your 4 source CSVs
│   ├── Combined Bond Indices Yields.csv
│   ├── EMB_data.csv
│   ├── Global finance data.csv
│   └── prices.csv
│
├── data_pipeline/
│   ├── zintellix_collector.py      ← fixed rewrite of the original Colab script
│   └── data_consolidator.py        ← CSV merge → unified_bond_master.csv
│
├── database/
│   ├── sqlite_store.py             ← structured store + indexed queries
│   ├── vector_store.py             ← ChromaDB + SentenceTransformers
│   ├── knowledge_graph.py          ← NetworkX MultiDiGraph
│   └── build_all.py                ← builds all three from unified CSV
│
├── engines/
│   ├── financial_engine.py         ← YTM / duration / convexity / spread
│   ├── search_engine.py            ← NL query parser + structured + semantic
│   ├── rag_engine.py               ← grounded copilot (OpenAI / Gemini)
│   ├── compliance_engine.py        ← restricted-phrase scrub + audit log
│   └── alerting_engine.py          ← yield / maturity / rating / liquidity alerts
│
├── api/
│   └── app.py                      ← Flask REST endpoints
│
├── ui/
│   └── streamlit_app.py            ← 8-page Streamlit dashboard
│
└── output/                         ← pre-built artefacts
    ├── unified_bond_master.csv     ← 58 unique ISINs
    ├── bond_platform.db            ← SQLite
    ├── knowledge_graph.gpickle     ← 124 nodes, 290 edges
    ├── chroma/                     ← ChromaDB vector store
    └── audit_log.jsonl             ← compliance audit trail
```

---

## 🚀 Run modes

```powershell
python run_all.py                 # full ingest + DB rebuild + launch
python run_all.py --no-live       # skip live NSE / zintellix calls (CSV-only)
python run_all.py --build-only    # rebuild DBs and exit
python run_all.py --serve-only    # skip rebuild, just launch services
```

To run the services **in separate terminal windows** (recommended for
seeing live logs from each independently):

```powershell
# Window 1 — Flask API
python -m api.app

# Window 2 — Streamlit UI
python -m streamlit run ui/streamlit_app.py
```

---

## 🤖 AI Copilot — getting the LLM working

The Copilot has two modes:

| Mode | When | What you see |
|---|---|---|
| **LLM-grounded** | API key present in `.env` | Gemini 2.5 Flash (or OpenAI GPT-4o-mini) generates a written, citation-backed answer |
| **Retrieval-only fallback** | No key, or LLM call fails | Deterministic stats synthesizer produces a structured comparison from the retrieved bonds. Still SEBI-safe — never hallucinates. |

Recommended: get a free **Gemini API key** at
<https://aistudio.google.com/apikey> and paste it into `.env`.

The default model is `gemini-2.5-flash`. Change `GEMINI_MODEL` in `.env`
to use a different one. **`gemini-1.5-flash` is deprecated** on the
v1beta endpoint — don't use it.

---

## 🧪 Try these queries in the AI Copilot

```
compare HDFC Ltd bonds vs ICICI Bank bonds
compare TCS Ltd vs Infosys Ltd
compare emerging market bonds vs corporate bonds
compare banking vs engineering bonds
AA bonds maturing within 3 years above 7% YTM
highest yielding PSU bonds
```

---

## 🔌 REST API summary

| Method | Path | Purpose |
|--------|------|---------|
| GET    | /health                       | liveness |
| GET    | /api/bonds                    | list (limit/offset) |
| GET    | /api/bonds/{isin}             | single bond |
| POST   | /api/search                   | natural-language search |
| POST   | /api/search/filter            | structured filter search |
| POST   | /api/analytics                | YTM / duration / convexity / spread |
| POST   | /api/copilot                  | RAG-grounded Q&A |
| GET    | /api/kg/issuer/{name}         | issuer exposure |
| GET    | /api/kg/sector/{name}         | sector → issuers |
| GET    | /api/kg/bond/{isin}           | bond neighbours |
| GET    | /api/alerts                   | run alert scan |
| GET    | /api/audit                    | recent audit log |
| GET    | /api/portfolio/summary        | counts + averages + breakdowns |

Example:

```powershell
curl http://localhost:5000/api/portfolio/summary

curl -X POST http://localhost:5000/api/copilot `
  -H "Content-Type: application/json" `
  -d '{"question":"compare HDFC vs ICICI bonds","k":6}'
```

---

## 🏗️ Architecture

```
LAYER 5  Application       Streamlit UI (8 pages) + Flask REST API
LAYER 4  AI                RAG Copilot (Gemini 2.5 / OpenAI) + retrieval fallback
LAYER 3  Intelligence      Financial engine (SciPy) + Knowledge Graph + Search
LAYER 2  ETL & Pipeline    zintellix collector + CSV consolidator
LAYER 1  Data              ChromaDB + SQLite + NetworkX KG
```

Mapped to the 17 features in the project brief:

| # | Feature | Where it lives |
|---|---|---|
| 1 | Bond Master Database | `data_pipeline/`, `database/sqlite_store.py` |
| 2 | Real-Time Bond Search Engine | `engines/search_engine.py` |
| 3 | Financial Analytics Engine | `engines/financial_engine.py` |
| 4 | AI Bond Copilot (RAG) | `engines/rag_engine.py` |
| 5 | Knowledge Graph Engine | `database/knowledge_graph.py` |
| 6 | Compliance & Governance | `engines/compliance_engine.py` |
| 7 | Secondary Market Intelligence | `data_pipeline/zintellix_collector.py` |
| 8 | Rating & Credit Intelligence | `engines/alerting_engine.py` |
| 9 | OCR & Document Intelligence | hook in Streamlit Admin page (extend with pdfplumber) |
| 10 | Portfolio Analytics Dashboard | Streamlit `Dashboard` page |
| 11 | AI Alerting & Monitoring | `engines/alerting_engine.py` + `Alerts` page |
| 12 | Workflow Automation | `compliance_engine.audit()` trails |
| 13 | Execution Management | API endpoints (extension point for OMS/EMS) |
| 14 | API & Integration Layer | `api/app.py` |
| 15 | Security Controls | env-only secrets + audit log |
| 16 | Admin / Governance Console | Streamlit `Admin` + `Compliance` pages |
| 17 | Data Pipeline & ETL | `data_pipeline/` |

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'chromadb'` | Wrong Python interpreter. Activate the venv first (`.\.venv\Scripts\Activate.ps1`) before `pip install`. |
| Streamlit port already in use | `Get-NetTCPConnection -LocalPort 8501 \| Stop-Process -Force` |
| `404 models/gemini-1.5-flash is not found` | Change `GEMINI_MODEL` in `.env` to `gemini-2.5-flash`. |
| Embeddings take a long time first time | One-time download of `all-MiniLM-L6-v2` (~80 MB). Cached after. |
| `pyarrow.lib.ArrowInvalid: Could not convert ''` | Already patched. If it recurs, run `python run_all.py --build-only`. |

---

## ⚠️ Disclaimer

This is a reference implementation for prototyping and demonstration.
It is **not** financial advice. Live execution / OMS / EMS integration
requires real broker credentials and SEBI-registered OBPP onboarding
before any production use.

---

**Built by:** Prabhhav / Zintellix Technologies
**Contact:** Prabhhav@zintellix.com
