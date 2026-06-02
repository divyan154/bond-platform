# """
# AI Bond Copilot — Retrieval-Augmented Generation.

#   1. Detect query intent (single-lookup vs comparison)
#   2. Retrieve grounded facts from ChromaDB + SQLite + Knowledge Graph
#        For comparisons, pull from BOTH sides of the comparison
#   3. Build a strict, citation-first context block
#   4. Ask an LLM (OpenAI or Gemini) to answer ONLY from that context
#      If no LLM key / call fails, an aggregate synthesizer produces
#      a real comparison answer from the retrieved data.

# Compliance:
#     * Restricted-phrase scrub (blocks "guaranteed return" etc.)
#     * Confidence score from top-hit cosine similarity
#     * Disclaimer always appended
# """
# from __future__ import annotations
# import re
# import sys
# import statistics
# from collections import Counter
# from pathlib import Path
# from typing import Optional

# sys.path.append(str(Path(__file__).resolve().parents[1]))
# from config import (OPENAI_API_KEY, GEMINI_API_KEY, OPENAI_MODEL, GEMINI_MODEL,
#                     LLM_PROVIDER, MIN_CONFIDENCE, COMPLIANCE_DISCLAIMER)
# from database.vector_store import semantic_search
# from database.sqlite_store  import fetch_by_isin, fetch_all
# from database.knowledge_graph import bond_neighbours, issuer_exposure
# from engines.compliance_engine import ComplianceEngine

# SYSTEM_PROMPT = """You are the Bond Intelligence Copilot for the Indian fixed-income market.
# You answer ONLY from the verified context provided below — never invent
# ISINs, ratings, yields, prices, or any number. If the context does not
# contain the answer, say so plainly. Cite each fact inline as [ISIN].
# Never use forbidden marketing terms like "guaranteed", "risk-free",
# or "assured return".
# For comparison questions, structure the answer as: definition of each
# category, side-by-side stats (counts, average coupon, average price,
# rating distribution), then a short analytical takeaway."""

# COMPARISON_RE = re.compile(
#     r"\b(compare|comparison|vs\.?|versus|difference between|differ)\b",
#     re.IGNORECASE,
# )

# CATEGORY_KEYWORDS = {
#     "emerging market":  ["emerging market", "emb", "emerging"],
#     "corporate":        ["corporate bond", "corporate"],
#     "government":       ["government", "g-sec", "gsec", "sovereign", "treasury"],
#     "psu":              ["psu", "public sector"],
#     "banking":          ["bank", "banking", "financial"],
#     "infrastructure":   ["infra", "infrastructure", "power", "energy"],
#     "it":               ["it ", "tech", "technology", "software"],
#     "engineering":      ["engineering"],
# }


# def _is_comparison(q: str) -> bool:
#     return bool(COMPARISON_RE.search(q))


# def _detect_categories(q: str) -> list[str]:
#     q_low = q.lower()
#     found: list[str] = []
#     for cat, keys in CATEGORY_KEYWORDS.items():
#         if any(k in q_low for k in keys):
#             found.append(cat)
#     return found


# def _bonds_matching_category(category: str, k: int = 10) -> list[dict]:
#     """Pull bonds whose BondType or Sector matches the category keyword."""
#     df = fetch_all()
#     if df.empty:
#         return []
#     pat = category.lower()
#     mask = (df["BondType"].astype(str).str.lower().str.contains(pat, na=False)
#             | df["Sector"].astype(str).str.lower().str.contains(pat, na=False))
#     df = df[mask].head(k)
#     return df.to_dict(orient="records")


# def _row_to_hit(row: dict) -> dict:
#     return {
#         "id":        row.get("ISIN"),
#         "document":  f"ISIN {row.get('ISIN')} | Issuer {row.get('IssuerName')} | "
#                      f"Sector {row.get('Sector')} | BondType {row.get('BondType')} | "
#                      f"Coupon {row.get('CouponRate')}% | Maturity {row.get('MaturityDate')}",
#         "metadata":  {k: row.get(k) for k in
#                       ("ISIN", "IssuerName", "Symbol", "Sector",
#                        "BondType", "Country", "Rating", "CouponRate",
#                        "MaturityDate", "LastTradedPrice", "Yield", "Source")},
#         "score":     None,
#         "full_row":  row,
#         "kg_neighbours": bond_neighbours(row.get("ISIN", "")) if row.get("ISIN") else {},
#         "issuer_exposure": issuer_exposure(row.get("IssuerName", "")) if row.get("IssuerName") else {},
#     }


# def _retrieve(query: str, k: int = 6) -> tuple[list[dict], float, dict]:
#     """Returns (hits, best_score, extras).

#     extras may include {'comparison': True, 'categories': [..], 'groups': {cat: [hits]}}
#     """
#     extras: dict = {}
#     if _is_comparison(query):
#         cats = _detect_categories(query)
#         if len(cats) >= 2:
#             extras["comparison"] = True
#             extras["categories"] = cats
#             extras["groups"] = {}
#             combined: list[dict] = []
#             seen_ids = set()
#             for cat in cats:
#                 rows = _bonds_matching_category(cat, k=k)
#                 group_hits = []
#                 for r in rows:
#                     if r["ISIN"] not in seen_ids:
#                         h = _row_to_hit(r)
#                         group_hits.append(h)
#                         combined.append(h)
#                         seen_ids.add(r["ISIN"])
#                 extras["groups"][cat] = group_hits
#             if combined:
#                 return combined, 0.9, extras

#     hits = semantic_search(query, k=k)
#     if not hits:
#         return [], 0.0, extras
#     best = max((h.get("score") or 0.0) for h in hits)
#     enriched = []
#     for h in hits:
#         isin = h["metadata"].get("ISIN")
#         row_df = fetch_by_isin(isin) if isin else None
#         row = row_df.iloc[0].to_dict() if (row_df is not None and not row_df.empty) else {}
#         issuer = row.get("IssuerName") or h["metadata"].get("IssuerName")
#         h["kg_neighbours"] = bond_neighbours(isin) if isin else {}
#         h["issuer_exposure"] = issuer_exposure(issuer) if issuer else {}
#         h["full_row"] = row
#         enriched.append(h)
#     return enriched, float(best), extras


# def _format_context(hits: list[dict]) -> str:
#     blocks = []
#     for h in hits:
#         m = h["metadata"]
#         full = h.get("full_row", {})
#         blocks.append(
#             f"[{m.get('ISIN')}]\n"
#             f"  Issuer        : {m.get('IssuerName') or full.get('IssuerName')}\n"
#             f"  Symbol        : {m.get('Symbol') or full.get('Symbol')}\n"
#             f"  Sector        : {m.get('Sector') or full.get('Sector')}\n"
#             f"  Country       : {m.get('Country') or full.get('Country')}\n"
#             f"  Bond Type     : {m.get('BondType') or full.get('BondType')}\n"
#             f"  Rating        : {m.get('Rating') or full.get('Rating')}\n"
#             f"  Coupon        : {m.get('CouponRate') or full.get('CouponRate')}%\n"
#             f"  Maturity      : {m.get('MaturityDate') or full.get('MaturityDate')}\n"
#             f"  Last Price    : {m.get('LastTradedPrice') or full.get('LastTradedPrice')}\n"
#             f"  Yield         : {m.get('Yield') or full.get('Yield')}\n"
#             f"  Source        : {m.get('Source') or full.get('Source')}\n"
#         )
#     return "\n".join(blocks) if blocks else "(no matching bonds)"


# def _llm_provider() -> Optional[str]:
#     if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
#         return "openai"
#     if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
#         return "gemini"
#     if LLM_PROVIDER == "auto":
#         if OPENAI_API_KEY:
#             return "openai"
#         if GEMINI_API_KEY:
#             return "gemini"
#     return None


# def _call_openai(context: str, question: str) -> str:
#     from openai import OpenAI
#     client = OpenAI(api_key=OPENAI_API_KEY)
#     resp = client.chat.completions.create(
#         model=OPENAI_MODEL,
#         temperature=0.0,
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content":
#                 f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"},
#         ],
#     )
#     return resp.choices[0].message.content.strip()


# def _call_gemini(context: str, question: str) -> str:
#     """Uses the new google-genai SDK with a current model."""
#     from google import genai
#     from google.genai import types
#     client = genai.Client(api_key=GEMINI_API_KEY)
#     resp = client.models.generate_content(
#         model=GEMINI_MODEL,
#         contents=f"CONTEXT:\n{context}\n\nQUESTION:\n{question}",
#         config=types.GenerateContentConfig(
#             system_instruction=SYSTEM_PROMPT,
#             temperature=0.0,
#         ),
#     )
#     return (resp.text or "").strip()


# def _safe_mean(xs: list) -> Optional[float]:
#     xs = [x for x in xs if isinstance(x, (int, float))]
#     return round(statistics.mean(xs), 3) if xs else None


# def _group_stats(hits: list[dict]) -> dict:
#     coupons = [h.get("full_row", {}).get("CouponRate") for h in hits]
#     prices  = [h.get("full_row", {}).get("LastTradedPrice") for h in hits]
#     yields  = [h.get("full_row", {}).get("Yield") for h in hits]
#     ratings = [h.get("full_row", {}).get("Rating") for h in hits if h.get("full_row", {}).get("Rating")]
#     sectors = [h.get("full_row", {}).get("Sector") for h in hits if h.get("full_row", {}).get("Sector")]
#     issuers = [h.get("full_row", {}).get("IssuerName") for h in hits if h.get("full_row", {}).get("IssuerName")]
#     return {
#         "count":              len(hits),
#         "avg_coupon_pct":     _safe_mean(coupons),
#         "avg_price":          _safe_mean(prices),
#         "avg_yield":          _safe_mean(yields),
#         "rating_distribution": dict(Counter(ratings).most_common()),
#         "sector_distribution": dict(Counter(sectors).most_common()),
#         "sample_issuers":     issuers[:5],
#     }


# def _retrieval_comparison_answer(extras: dict, question: str) -> str:
#     """Aggregate-stats comparison synthesized purely from retrieved rows."""
#     cats = extras.get("categories", [])
#     groups = extras.get("groups", {})

#     lines = [f"### Comparison: {' vs '.join(c.title() for c in cats)}\n"]
#     lines.append(f"_Query: \"{question}\". Synthesised from {sum(len(v) for v in groups.values())} retrieved bond records._\n")

#     for cat in cats:
#         hits = groups.get(cat, [])
#         if not hits:
#             lines.append(f"**{cat.title()}** — no matching bonds in the corpus.\n")
#             continue
#         s = _group_stats(hits)
#         lines.append(f"**{cat.title()}** ({s['count']} bonds)")
#         lines.append(f"- Avg coupon : {s['avg_coupon_pct']}%")
#         lines.append(f"- Avg price  : {s['avg_price']}")
#         lines.append(f"- Avg yield  : {s['avg_yield']}")
#         lines.append(f"- Ratings    : {s['rating_distribution']}")
#         lines.append(f"- Sectors    : {s['sector_distribution']}")
#         lines.append(f"- Issuers    : {', '.join(s['sample_issuers'])}")
#         lines.append(f"- Sample ISINs: {', '.join(h['id'] for h in hits[:3])}\n")

#     stats_per_cat = {c: _group_stats(groups.get(c, [])) for c in cats}
#     valid = {c: s for c, s in stats_per_cat.items()
#              if s["count"] and s["avg_coupon_pct"] is not None}
#     if len(valid) >= 2:
#         cats_sorted = sorted(valid.items(),
#                              key=lambda kv: kv[1]["avg_coupon_pct"] or 0,
#                              reverse=True)
#         hi, lo = cats_sorted[0], cats_sorted[-1]
#         diff = round((hi[1]["avg_coupon_pct"] or 0) - (lo[1]["avg_coupon_pct"] or 0), 2)
#         lines.append("**Takeaway**")
#         lines.append(
#             f"- {hi[0].title()} bonds carry the higher average coupon "
#             f"({hi[1]['avg_coupon_pct']}%) vs {lo[0].title()} "
#             f"({lo[1]['avg_coupon_pct']}%) — a spread of {diff} percentage points."
#         )
#         lines.append(
#             "- Rating mix and issuer profile drive the spread: "
#             f"{hi[0].title()} skews to {', '.join(list(hi[1]['rating_distribution'].keys())[:2]) or 'mixed'} "
#             f"ratings while {lo[0].title()} skews to "
#             f"{', '.join(list(lo[1]['rating_distribution'].keys())[:2]) or 'mixed'}."
#         )

#     return "\n".join(lines)


# def _retrieval_listing_answer(hits: list[dict], question: str) -> str:
#     if not hits:
#         return "No matching bonds found in the verified data sources."
#     lines = [f"Top {len(hits)} matches for: \"{question}\"\n"]
#     for h in hits:
#         m = h["metadata"]
#         lines.append(
#             f"- [{m.get('ISIN')}] {m.get('IssuerName') or m.get('Symbol')} | "
#             f"{m.get('Sector') or 'N/A'} | {m.get('BondType') or 'N/A'} | "
#             f"Rating {m.get('Rating') or 'N/A'} | Coupon {m.get('CouponRate') or 'N/A'}% | "
#             f"Maturity {m.get('MaturityDate') or 'N/A'} | "
#             f"Last Px {m.get('LastTradedPrice') or 'N/A'} (source: {m.get('Source')})"
#         )
#     return "\n".join(lines)


# def _retrieval_only_answer(hits: list[dict], extras: dict, question: str) -> str:
#     if extras.get("comparison"):
#         return _retrieval_comparison_answer(extras, question)
#     return _retrieval_listing_answer(hits, question)


# def ask(question: str, k: int = 6) -> dict:
#     if not question or not question.strip():
#         return {"answer": "Please provide a question.", "citations": [],
#                 "confidence": 0.0, "blocked": False}

#     hits, confidence, extras = _retrieve(question, k=k)
#     context = _format_context(hits)
#     provider = _llm_provider()
#     used = "retrieval-only"
#     try:
#         if provider == "openai":
#             raw_answer = _call_openai(context, question); used = f"openai/{OPENAI_MODEL}"
#         elif provider == "gemini":
#             raw_answer = _call_gemini(context, question); used = f"gemini/{GEMINI_MODEL}"
#         else:
#             raw_answer = _retrieval_only_answer(hits, extras, question)
#     except Exception as e:
#         raw_answer = _retrieval_only_answer(hits, extras, question)
#         used = f"retrieval-only (LLM failed: {e})"

#     scrubbed, violations = ComplianceEngine.scrub(raw_answer)
#     blocked = (confidence < MIN_CONFIDENCE
#                and used.startswith("retrieval-only")
#                and not hits
#                and not extras.get("comparison"))

#     answer = scrubbed
#     if blocked:
#         answer = ("Insufficient verified evidence to answer this question safely. "
#                   "Please refine the query or upload supporting documents.")
#     answer += "\n\n" + COMPLIANCE_DISCLAIMER

#     return {
#         "question":         question,
#         "answer":           answer,
#         "citations":        [h["metadata"].get("ISIN") for h in hits],
#         "evidence":         hits,
#         "confidence":       round(confidence, 3),
#         "engine":           used,
#         "blocked":          blocked,
#         "comparison_mode":  extras.get("comparison", False),
#         "categories":       extras.get("categories", []),
#         "compliance_flags": violations,
#     }


# if __name__ == "__main__":
#     import json
#     print(json.dumps(ask("compare emerging bond vs corporate bond"),
#                      indent=2, default=str))
"""
AI Bond Copilot — Retrieval-Augmented Generation.

  1. Detect query intent (single-lookup vs comparison)
  2. Retrieve grounded facts from ChromaDB + SQLite + Knowledge Graph
       For comparisons, pull from BOTH sides of the comparison
  3. Build a strict, citation-first context block
  4. Ask an LLM (OpenAI or Gemini) to answer ONLY from that context
     If no LLM key / call fails, an aggregate synthesizer produces
     a real comparison answer from the retrieved data.

Compliance:
    * Restricted-phrase scrub (blocks "guaranteed return" etc.)
    * Confidence score from top-hit cosine similarity
    * Disclaimer always appended
"""
from __future__ import annotations
import re
import sys
import statistics
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (OPENAI_MODEL, GEMINI_MODEL, MIN_CONFIDENCE, COMPLIANCE_DISCLAIMER)
from database.vector_store import semantic_search
from database.sqlite_store  import fetch_by_isin, fetch_all
from database.knowledge_graph import bond_neighbours, issuer_exposure
from engines.compliance_engine import ComplianceEngine

SYSTEM_PROMPT = """You are a senior fixed-income analyst and bond-market educator
specialising in the Indian debt capital market (with global comparison
capability). Your job is to give COMPLETE, EXPERT-QUALITY answers — not
just dump retrieved rows.

You operate under a strict separation between FACTS and KNOWLEDGE:

══════════════════ NUMERICAL / FACTUAL CLAIMS ══════════════════
Any SPECIFIC number tied to a SPECIFIC bond — yield, coupon, price,
rating, maturity date, ISIN, volume, spread, duration, issuer-level
exposure — MUST come from the CORPUS DATA / CORPUS SUMMARY sections
below. Cite the source ISIN inline in square brackets: [INE001A07EM1].
Never invent these. If the corpus lacks a specific figure, say so.

══════════════════ GENERAL KNOWLEDGE & EXPERTISE ══════════════════
You CAN and SHOULD use your professional expertise freely on:
  • Bond market concepts (YTM, Macaulay/modified duration, convexity,
    accrual basis, day-count conventions, dirty vs clean price...)
  • Financial theory and formulas (Fisher equation, term structure,
    DV01, key-rate duration, credit spreads, OAS...)
  • Bond type definitions (sovereign, G-Sec, T-Bills, SDL, PSU, AT1,
    perpetual, callable, puttable, zero-coupon, EMB, IG, HY...)
  • Market dynamics (credit risk, interest-rate risk, reinvestment risk,
    liquidity risk, inflation-linked instruments...)
  • Indian-market specifics (SEBI OBPP framework, RBI auctions, NDS-OM,
    settlement T+1, listed vs unlisted, ISIN structure, rating agencies)
  • Methodology (how to read a yield curve, how spreads decompose,
    how to compare bonds across ratings/maturities, portfolio laddering)
General knowledge does NOT need citations — it is professional expertise.

══════════════════ DEFAULT ANSWER STRUCTURE ══════════════════
Unless the user asks for a one-line answer, structure as:

  1. **Direct answer** (2–4 sentences). State the conclusion up front.
  2. **What the data shows** — pull relevant rows from CORPUS DATA with
     [ISIN] citations. Use a table for comparisons.
  3. **Concept / methodology** — explain the underlying finance
     (definitions, formulas, risk drivers) using your expertise.
  4. **Analytical takeaway** — what this means for an investor /
     portfolio manager. Mention key risks.

══════════════════ WHEN THE CORPUS IS THIN ══════════════════
Do NOT refuse. Do this instead:
  • Answer the conceptual question using your expertise
  • State plainly: "The loaded corpus contains [N] matching bonds, so I
    cannot quote live figures. Conceptually, …"
  • Offer guidance on what data would let you give a quantitative answer
    (e.g. NSDL ISIN lookup, NSE DEBT segment, rating agency filings)

══════════════════ HARD GUARDRAILS ══════════════════
  • NEVER use marketing terms: "guaranteed", "risk-free", "assured
    return", "100% safe", "sure shot".
  • NEVER fabricate ISINs, ratings, prices, or yields.
  • NEVER provide investment advice for a specific person.
  • Acknowledge uncertainty. Quantitative outputs are advisory.
  • If asked about regulation, defer to SEBI / RBI as the authority.

Be precise, analytical, and useful. Write like a fixed-income desk
strategist briefing a portfolio manager — not like a search engine."""

COMPARISON_RE = re.compile(
    r"\b(compare|comparison|vs\.?|versus|difference between|differ)\b",
    re.IGNORECASE,
)

CATEGORY_KEYWORDS = {
    "emerging market":  ["emerging market", "emb", "emerging"],
    "corporate":        ["corporate bond", "corporate"],
    "government":       ["government", "g-sec", "gsec", "sovereign", "treasury"],
    "psu":              ["psu", "public sector"],
    "banking":          ["bank", "banking", "financial"],
    "infrastructure":   ["infra", "infrastructure", "power", "energy"],
    "it":               ["it ", "tech", "technology", "software"],
    "engineering":      ["engineering"],
}


def _is_comparison(q: str) -> bool:
    return bool(COMPARISON_RE.search(q))


def _detect_categories(q: str) -> list[str]:
    q_low = q.lower()
    found: list[str] = []
    for cat, keys in CATEGORY_KEYWORDS.items():
        if any(k in q_low for k in keys):
            found.append(cat)
    return found


def _bonds_matching_category(category: str, k: int = 10) -> list[dict]:
    """Pull bonds whose BondType or Sector matches the category keyword."""
    df = fetch_all()
    if df.empty:
        return []
    pat = category.lower()
    mask = (df["BondType"].astype(str).str.lower().str.contains(pat, na=False)
            | df["Sector"].astype(str).str.lower().str.contains(pat, na=False))
    df = df[mask].head(k)
    return df.to_dict(orient="records")


def _row_to_hit(row: dict) -> dict:
    return {
        "id":        row.get("ISIN"),
        "document":  f"ISIN {row.get('ISIN')} | Issuer {row.get('IssuerName')} | "
                     f"Sector {row.get('Sector')} | BondType {row.get('BondType')} | "
                     f"Coupon {row.get('CouponRate')}% | Maturity {row.get('MaturityDate')}",
        "metadata":  {k: row.get(k) for k in
                      ("ISIN", "IssuerName", "Symbol", "Sector",
                       "BondType", "Country", "Rating", "CouponRate",
                       "MaturityDate", "LastTradedPrice", "Yield", "Source")},
        "score":     None,
        "full_row":  row,
        "kg_neighbours": bond_neighbours(row.get("ISIN", "")) if row.get("ISIN") else {},
        "issuer_exposure": issuer_exposure(row.get("IssuerName", "")) if row.get("IssuerName") else {},
    }


def _retrieve(query: str, k: int = 6) -> tuple[list[dict], float, dict]:
    """Returns (hits, best_score, extras).

    extras may include {'comparison': True, 'categories': [..], 'groups': {cat: [hits]}}
    """
    extras: dict = {}
    if _is_comparison(query):
        cats = _detect_categories(query)
        if len(cats) >= 2:
            extras["comparison"] = True
            extras["categories"] = cats
            extras["groups"] = {}
            combined: list[dict] = []
            seen_ids = set()
            for cat in cats:
                rows = _bonds_matching_category(cat, k=k)
                group_hits = []
                for r in rows:
                    if r["ISIN"] not in seen_ids:
                        h = _row_to_hit(r)
                        group_hits.append(h)
                        combined.append(h)
                        seen_ids.add(r["ISIN"])
                extras["groups"][cat] = group_hits
            if combined:
                return combined, 0.9, extras

    hits = semantic_search(query, k=k)
    if not hits:
        return [], 0.0, extras
    best = max((h.get("score") or 0.0) for h in hits)
    enriched = []
    for h in hits:
        isin = h["metadata"].get("ISIN")
        row_df = fetch_by_isin(isin) if isin else None
        row = row_df.iloc[0].to_dict() if (row_df is not None and not row_df.empty) else {}
        issuer = row.get("IssuerName") or h["metadata"].get("IssuerName")
        h["kg_neighbours"] = bond_neighbours(isin) if isin else {}
        h["issuer_exposure"] = issuer_exposure(issuer) if issuer else {}
        h["full_row"] = row
        enriched.append(h)
    return enriched, float(best), extras


def _format_context(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        m = h["metadata"]
        full = h.get("full_row", {})
        blocks.append(
            f"[{m.get('ISIN')}]\n"
            f"  Issuer        : {m.get('IssuerName') or full.get('IssuerName')}\n"
            f"  Symbol        : {m.get('Symbol') or full.get('Symbol')}\n"
            f"  Sector        : {m.get('Sector') or full.get('Sector')}\n"
            f"  Country       : {m.get('Country') or full.get('Country')}\n"
            f"  Bond Type     : {m.get('BondType') or full.get('BondType')}\n"
            f"  Rating        : {m.get('Rating') or full.get('Rating')}\n"
            f"  Coupon        : {m.get('CouponRate') or full.get('CouponRate')}%\n"
            f"  Maturity      : {m.get('MaturityDate') or full.get('MaturityDate')}\n"
            f"  Last Price    : {m.get('LastTradedPrice') or full.get('LastTradedPrice')}\n"
            f"  Yield         : {m.get('Yield') or full.get('Yield')}\n"
            f"  Source        : {m.get('Source') or full.get('Source')}\n"
        )
    return "\n".join(blocks) if blocks else "(no specific bonds retrieved for this query)"


def _corpus_snapshot() -> str:
    """Compact summary of the whole Bond Master so the LLM always knows
    what's loaded — even when retrieval returns nothing close."""
    try:
        df = fetch_all()
    except Exception:
        return "(corpus snapshot unavailable)"
    if df.empty:
        return "(no bonds loaded in the database)"

    df_clean = df.copy()
    df_clean["CouponRate"] = __import__("pandas").to_numeric(
        df_clean.get("CouponRate"), errors="coerce")
    df_clean["LastTradedPrice"] = __import__("pandas").to_numeric(
        df_clean.get("LastTradedPrice"), errors="coerce")

    def _vc(col, n=8):
        s = df_clean[col].astype(str).replace({"nan": "", "None": ""}).str.strip()
        s = s[s != ""]
        return {str(k): int(v) for k, v in s.value_counts().head(n).items()}

    coupon_series = df_clean["CouponRate"].dropna()
    coupon_series = coupon_series[coupon_series > 0]
    price_series  = df_clean["LastTradedPrice"].dropna()
    price_series  = price_series[price_series > 0]

    avg_coupon = coupon_series.mean() if not coupon_series.empty else None
    avg_price  = price_series.mean()  if not price_series.empty  else None
    has_meta_count = int(df_clean["BondType"].astype(str).str.strip().replace(
        {"nan": "", "None": ""}).ne("").sum())

    issuers = sorted(set(
        str(x).strip() for x in df["IssuerName"].dropna()
        if str(x).strip() and "Rated Bonds" not in str(x)
    ))[:25]

    parts = [
        f"Total bonds loaded     : {len(df)}",
        f"Bonds with full metadata (coupon/maturity/type): {has_meta_count}",
        f"Unique issuers         : {df['IssuerName'].nunique()}",
        f"Avg coupon (non-zero)  : {avg_coupon:.2f}%" if avg_coupon is not None else "Avg coupon: N/A",
        f"Avg last price (non-zero): {avg_price:.2f}" if avg_price  is not None else "Avg last price: N/A",
        f"Rating distribution    : {_vc('Rating')}",
        f"Sector distribution    : {_vc('Sector')}",
        f"Country distribution   : {_vc('Country')}",
        f"BondType distribution  : {_vc('BondType')}",
        f"Sample issuers         : {', '.join(issuers[:15])}",
    ]
    return "\n".join(parts)


def _build_prompt(context: str, snapshot: str, question: str) -> str:
    return (
        "CORPUS SUMMARY (whole database snapshot — use this for "
        "questions about counts, distribution, sectors, issuers):\n"
        f"{snapshot}\n\n"
        "CORPUS DATA (most relevant bonds for this query — cite these "
        "ISINs when quoting numbers):\n"
        f"{context}\n\n"
        "USER QUESTION:\n"
        f"{question}\n\n"
        "Answer following the structure in your system instructions. "
        "Use your professional expertise freely for concepts, theory, "
        "definitions, and methodology. Only cite ISINs for specific "
        "numerical claims about specific bonds in the corpus."
    )


def _llm_provider() -> Optional[str]:
    import os
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    provider = os.getenv("LLM_PROVIDER", "auto").lower()
    if provider == "openai" and openai_key:
        return "openai"
    if provider == "gemini" and gemini_key:
        return "gemini"
    if provider == "auto":
        if openai_key:
            return "openai"
        if gemini_key:
            return "gemini"
    return None


def _call_openai(context: str, snapshot: str, question: str) -> str:
    import os
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(context, snapshot, question)},
        ],
    )
    return resp.choices[0].message.content.strip()


def _call_gemini(context: str, snapshot: str, question: str) -> str:
    """Uses the new google-genai SDK with a current model."""
    import os
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_build_prompt(context, snapshot, question),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )
    return (resp.text or "").strip()


def _safe_mean(xs: list) -> Optional[float]:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.mean(xs), 3) if xs else None


def _group_stats(hits: list[dict]) -> dict:
    coupons = [h.get("full_row", {}).get("CouponRate") for h in hits]
    prices  = [h.get("full_row", {}).get("LastTradedPrice") for h in hits]
    yields  = [h.get("full_row", {}).get("Yield") for h in hits]
    ratings = [h.get("full_row", {}).get("Rating") for h in hits if h.get("full_row", {}).get("Rating")]
    sectors = [h.get("full_row", {}).get("Sector") for h in hits if h.get("full_row", {}).get("Sector")]
    issuers = [h.get("full_row", {}).get("IssuerName") for h in hits if h.get("full_row", {}).get("IssuerName")]
    return {
        "count":              len(hits),
        "avg_coupon_pct":     _safe_mean(coupons),
        "avg_price":          _safe_mean(prices),
        "avg_yield":          _safe_mean(yields),
        "rating_distribution": dict(Counter(ratings).most_common()),
        "sector_distribution": dict(Counter(sectors).most_common()),
        "sample_issuers":     issuers[:5],
    }


def _retrieval_comparison_answer(extras: dict, question: str) -> str:
    """Aggregate-stats comparison synthesized purely from retrieved rows."""
    cats = extras.get("categories", [])
    groups = extras.get("groups", {})

    lines = [f"### Comparison: {' vs '.join(c.title() for c in cats)}\n"]
    lines.append(f"_Query: \"{question}\". Synthesised from {sum(len(v) for v in groups.values())} retrieved bond records._\n")

    for cat in cats:
        hits = groups.get(cat, [])
        if not hits:
            lines.append(f"**{cat.title()}** — no matching bonds in the corpus.\n")
            continue
        s = _group_stats(hits)
        lines.append(f"**{cat.title()}** ({s['count']} bonds)")
        lines.append(f"- Avg coupon : {s['avg_coupon_pct']}%")
        lines.append(f"- Avg price  : {s['avg_price']}")
        lines.append(f"- Avg yield  : {s['avg_yield']}")
        lines.append(f"- Ratings    : {s['rating_distribution']}")
        lines.append(f"- Sectors    : {s['sector_distribution']}")
        lines.append(f"- Issuers    : {', '.join(s['sample_issuers'])}")
        lines.append(f"- Sample ISINs: {', '.join(h['id'] for h in hits[:3])}\n")

    stats_per_cat = {c: _group_stats(groups.get(c, [])) for c in cats}
    valid = {c: s for c, s in stats_per_cat.items()
             if s["count"] and s["avg_coupon_pct"] is not None}
    if len(valid) >= 2:
        cats_sorted = sorted(valid.items(),
                             key=lambda kv: kv[1]["avg_coupon_pct"] or 0,
                             reverse=True)
        hi, lo = cats_sorted[0], cats_sorted[-1]
        diff = round((hi[1]["avg_coupon_pct"] or 0) - (lo[1]["avg_coupon_pct"] or 0), 2)
        lines.append("**Takeaway**")
        lines.append(
            f"- {hi[0].title()} bonds carry the higher average coupon "
            f"({hi[1]['avg_coupon_pct']}%) vs {lo[0].title()} "
            f"({lo[1]['avg_coupon_pct']}%) — a spread of {diff} percentage points."
        )
        lines.append(
            "- Rating mix and issuer profile drive the spread: "
            f"{hi[0].title()} skews to {', '.join(list(hi[1]['rating_distribution'].keys())[:2]) or 'mixed'} "
            f"ratings while {lo[0].title()} skews to "
            f"{', '.join(list(lo[1]['rating_distribution'].keys())[:2]) or 'mixed'}."
        )

    return "\n".join(lines)


def _retrieval_listing_answer(hits: list[dict], question: str) -> str:
    if not hits:
        return "No matching bonds found in the verified data sources."
    lines = [f"Top {len(hits)} matches for: \"{question}\"\n"]
    for h in hits:
        m = h["metadata"]
        lines.append(
            f"- [{m.get('ISIN')}] {m.get('IssuerName') or m.get('Symbol')} | "
            f"{m.get('Sector') or 'N/A'} | {m.get('BondType') or 'N/A'} | "
            f"Rating {m.get('Rating') or 'N/A'} | Coupon {m.get('CouponRate') or 'N/A'}% | "
            f"Maturity {m.get('MaturityDate') or 'N/A'} | "
            f"Last Px {m.get('LastTradedPrice') or 'N/A'} (source: {m.get('Source')})"
        )
    return "\n".join(lines)


def _retrieval_only_answer(hits: list[dict], extras: dict, question: str) -> str:
    if extras.get("comparison"):
        return _retrieval_comparison_answer(extras, question)
    return _retrieval_listing_answer(hits, question)


def ask(question: str, k: int = 10) -> dict:
    if not question or not question.strip():
        return {"answer": "Please provide a question.", "citations": [],
                "confidence": 0.0, "blocked": False}

    hits, confidence, extras = _retrieve(question, k=k)
    context  = _format_context(hits)
    snapshot = _corpus_snapshot()
    provider = _llm_provider()
    used = "retrieval-only"
    try:
        if provider == "openai":
            raw_answer = _call_openai(context, snapshot, question)
            used = f"openai/{OPENAI_MODEL}"
        elif provider == "gemini":
            raw_answer = _call_gemini(context, snapshot, question)
            used = f"gemini/{GEMINI_MODEL}"
        else:
            raw_answer = _retrieval_only_answer(hits, extras, question)
    except Exception as e:
        raw_answer = _retrieval_only_answer(hits, extras, question)
        used = f"retrieval-only (LLM failed: {e})"

    scrubbed, violations = ComplianceEngine.scrub(raw_answer)
    # LLM-grounded responses are NEVER blocked on confidence — the LLM
    # is trusted to say "I don't know" itself. Only block when retrieval
    # fallback has truly nothing to say.
    blocked = (used.startswith("retrieval-only")
               and not hits
               and not extras.get("comparison")
               and confidence < MIN_CONFIDENCE)

    answer = scrubbed
    if blocked:
        answer = ("Insufficient verified evidence to answer this question safely. "
                  "Please refine the query or upload supporting documents.")
    answer += "\n\n" + COMPLIANCE_DISCLAIMER

    return {
        "question":         question,
        "answer":           answer,
        "citations":        [h["metadata"].get("ISIN") for h in hits],
        "evidence":         hits,
        "confidence":       round(confidence, 3),
        "engine":           used,
        "blocked":          blocked,
        "comparison_mode":  extras.get("comparison", False),
        "categories":       extras.get("categories", []),
        "compliance_flags": violations,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(ask("compare emerging bond vs corporate bond"),
                     indent=2, default=str))
