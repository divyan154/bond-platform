"""
SEBI-aligned compliance & governance engine.

* Scrubs restricted marketing phrases from AI output
* Maintains an audit trail
* Provides suitability checks (rating floor, ticket size, sector cap)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RESTRICTED_PHRASES, OUTPUT_DIR

AUDIT_LOG = OUTPUT_DIR / "audit_log.jsonl"


class ComplianceEngine:
    @staticmethod
    def scrub(text: str) -> tuple[str, list[str]]:
        violations = []
        scrubbed = text
        for phrase in RESTRICTED_PHRASES:
            if phrase.lower() in scrubbed.lower():
                violations.append(phrase)
                scrubbed = scrubbed.replace(phrase, "[REDACTED-COMPLIANCE]")
                scrubbed = scrubbed.replace(phrase.title(), "[REDACTED-COMPLIANCE]")
                scrubbed = scrubbed.replace(phrase.upper(), "[REDACTED-COMPLIANCE]")
        return scrubbed, violations

    @staticmethod
    def suitability_check(user_profile: dict, bond_row: dict) -> dict:
        """Returns {ok: bool, reasons: [...] }."""
        reasons = []
        min_rating = user_profile.get("min_rating", "BBB").upper()
        rating_order = ["D", "C", "B", "BB", "BBB", "A", "AA", "AAA"]
        try:
            bond_rating = str(bond_row.get("Rating") or "BBB").upper()
            base = next((r for r in rating_order if bond_rating.startswith(r)), "BBB")
            if rating_order.index(base) < rating_order.index(min_rating):
                reasons.append(
                    f"Bond rating {bond_rating} below user floor {min_rating}"
                )
        except Exception:
            pass

        sector_cap = user_profile.get("sector_cap", {})
        sector = str(bond_row.get("Sector") or "")
        if sector and sector in sector_cap and sector_cap[sector] <= 0:
            reasons.append(f"Sector {sector} fully utilised in user portfolio")

        return {"ok": not reasons, "reasons": reasons}

    @staticmethod
    def audit(actor: str, action: str, payload: dict) -> None:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts":      datetime.utcnow().isoformat() + "Z",
            "actor":   actor,
            "action":  action,
            "payload": payload,
        }
        with AUDIT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    def recent_audit(n: int = 20) -> list[dict]:
        if not AUDIT_LOG.exists():
            return []
        with AUDIT_LOG.open("r", encoding="utf-8") as f:
            lines = f.readlines()[-n:]
        return [json.loads(l) for l in lines if l.strip()]
