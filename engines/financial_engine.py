"""
Deterministic Fixed Income Analytics Engine.

The PDF brief calls this the "truth engine". Pure-math, no AI:
    * Yield to Maturity         (Brent root-finder)
    * Clean & dirty price       (PV of cashflows)
    * Accrued interest          (30/360 actual basis)
    * Macaulay & modified duration
    * Convexity
    * Spread vs benchmark
    * Risk-return score
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np
from scipy.optimize import brentq


def _to_date(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except Exception:
        try:
            return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
        except Exception:
            return None


@dataclass
class Bond:
    face_value:    float = 1000.0
    coupon_rate:   float = 0.0        # annual % (e.g. 7.5)
    frequency:     int   = 2          # coupon payments per year
    issue_date:    Optional[date] = None
    maturity_date: Optional[date] = None
    settlement:    Optional[date] = None  # default = today

    @classmethod
    def from_row(cls, row: dict) -> "Bond":
        return cls(
            face_value    = float(row.get("FaceValue") or 1000.0),
            coupon_rate   = float(row.get("CouponRate") or 0.0),
            frequency     = int(row.get("Frequency") or 2),
            issue_date    = _to_date(row.get("IssueDate")),
            maturity_date = _to_date(row.get("MaturityDate")),
            settlement    = _to_date(row.get("Settlement")) or date.today(),
        )

    def cashflows(self) -> list[tuple[float, float]]:
        """Returns [(years_from_settle, cashflow), ...]"""
        if not self.maturity_date:
            return []
        settle = self.settlement or date.today()
        if self.maturity_date <= settle:
            return []
        coupon = self.face_value * (self.coupon_rate / 100) / self.frequency
        period_years = 1 / self.frequency
        n = max(1, int(np.ceil(((self.maturity_date - settle).days / 365.25)
                               * self.frequency)))
        cfs = []
        for i in range(1, n + 1):
            t = i * period_years
            cf = coupon + (self.face_value if i == n else 0)
            cfs.append((t, cf))
        return cfs


@dataclass
class AnalyticsResult:
    ytm:               Optional[float] = None      # decimal e.g. 0.0742
    clean_price:       Optional[float] = None
    dirty_price:       Optional[float] = None
    accrued_interest:  Optional[float] = None
    macaulay_duration: Optional[float] = None
    modified_duration: Optional[float] = None
    convexity:         Optional[float] = None
    spread_bps:        Optional[float] = None
    risk_return_score: Optional[float] = None
    notes:             list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in self.__dict__.items() if k != "notes"} | {"notes": self.notes}


class FinancialEngine:
    @staticmethod
    def price_from_yield(bond: Bond, ytm: float) -> float:
        return sum(cf / (1 + ytm / bond.frequency) ** (t * bond.frequency)
                   for t, cf in bond.cashflows())

    @staticmethod
    def yield_to_maturity(bond: Bond, market_price: float) -> Optional[float]:
        cfs = bond.cashflows()
        if not cfs or market_price <= 0:
            return None

        def f(y):
            return FinancialEngine.price_from_yield(bond, y) - market_price

        try:
            return brentq(f, -0.5, 5.0, maxiter=200, xtol=1e-8)
        except Exception:
            return None

    @staticmethod
    def accrued_interest(bond: Bond) -> float:
        if not bond.issue_date or not bond.maturity_date:
            return 0.0
        settle = bond.settlement or date.today()
        period_days = 365.25 / bond.frequency
        since_issue_days = (settle - bond.issue_date).days
        if since_issue_days <= 0:
            return 0.0
        frac_period = (since_issue_days % period_days) / period_days
        coupon = bond.face_value * (bond.coupon_rate / 100) / bond.frequency
        return coupon * frac_period

    @staticmethod
    def duration_convexity(bond: Bond, ytm: float) -> tuple[float, float, float]:
        cfs = bond.cashflows()
        if not cfs:
            return 0.0, 0.0, 0.0
        pv_total = 0.0
        weighted_t = 0.0
        convex = 0.0
        for t, cf in cfs:
            pv = cf / (1 + ytm / bond.frequency) ** (t * bond.frequency)
            pv_total += pv
            weighted_t += t * pv
            convex += t * (t + 1 / bond.frequency) * pv
        if pv_total <= 0:
            return 0.0, 0.0, 0.0
        macaulay = weighted_t / pv_total
        modified = macaulay / (1 + ytm / bond.frequency)
        convexity = convex / (pv_total * (1 + ytm / bond.frequency) ** 2)
        return macaulay, modified, convexity

    @classmethod
    def analyse(cls, row: dict,
                benchmark_yield: Optional[float] = None) -> AnalyticsResult:
        bond = Bond.from_row(row)
        res = AnalyticsResult()

        if not bond.maturity_date:
            res.notes.append("Missing maturity date — analytics skipped")
            return res

        market_price = row.get("LastTradedPrice") or row.get("Price")
        try:
            market_price = float(market_price) if market_price else None
        except Exception:
            market_price = None

        if market_price:
            res.ytm = cls.yield_to_maturity(bond, market_price)
            res.dirty_price = market_price
            res.accrued_interest = cls.accrued_interest(bond)
            res.clean_price = market_price - res.accrued_interest
        else:
            res.notes.append("No market price — YTM computed from coupon estimate")
            res.ytm = bond.coupon_rate / 100 if bond.coupon_rate else None

        if res.ytm is not None:
            mac, mod, conv = cls.duration_convexity(bond, res.ytm)
            res.macaulay_duration = mac
            res.modified_duration = mod
            res.convexity         = conv

        if res.ytm is not None and benchmark_yield is not None:
            res.spread_bps = (res.ytm - benchmark_yield) * 10000

        if res.ytm is not None and res.modified_duration is not None:
            res.risk_return_score = float(
                np.tanh((res.ytm * 100) / max(1.0, res.modified_duration))
            )

        return res


if __name__ == "__main__":
    sample = {
        "FaceValue": 1000, "CouponRate": 7.5, "IssueDate": "2022-01-01",
        "MaturityDate": "2029-01-01", "LastTradedPrice": 985.0,
    }
    print(FinancialEngine.analyse(sample, benchmark_yield=0.07).as_dict())
