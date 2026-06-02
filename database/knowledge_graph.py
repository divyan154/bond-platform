"""
Knowledge Graph (NetworkX) for issuer / sector / rating / bond relationships.

Nodes:
    bond:<ISIN>        Bond
    issuer:<name>      Issuer
    sector:<name>      Sector
    rating:<grade>     Credit rating
    country:<name>     Country
    bondtype:<name>    Bond type bucket

Edges:
    bond -> issuer    (issued_by)
    bond -> rating    (rated)
    issuer -> sector  (operates_in)
    issuer -> country (domiciled_in)
    bond -> bondtype  (instrument_of)
"""
from __future__ import annotations
import sys
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import KG_FILE, UNIFIED_CSV

_graph: nx.MultiDiGraph | None = None


def _node(kind: str, value: str) -> str:
    return f"{kind}:{str(value).strip()}"


def build_graph(df: pd.DataFrame | None = None) -> nx.MultiDiGraph:
    if df is None:
        df = pd.read_csv(UNIFIED_CSV)
    G = nx.MultiDiGraph()

    for _, r in df.iterrows():
        isin = str(r.get("ISIN", "")).strip()
        if not isin:
            continue
        issuer  = (str(r.get("IssuerName", "")).strip() or
                   str(r.get("Symbol", "")).strip() or "Unknown")
        sector  = str(r.get("Sector", "")).strip() or "Unspecified"
        rating  = str(r.get("Rating", "")).strip() or "Unrated"
        country = str(r.get("Country", "")).strip() or "India"
        btype   = str(r.get("BondType", "")).strip() or "Corporate"

        n_bond    = _node("bond", isin)
        n_issuer  = _node("issuer", issuer)
        n_sector  = _node("sector", sector)
        n_rating  = _node("rating", rating)
        n_country = _node("country", country)
        n_btype   = _node("bondtype", btype)

        G.add_node(n_bond, kind="bond", isin=isin,
                   symbol=str(r.get("Symbol", "")),
                   coupon=r.get("CouponRate"),
                   maturity=str(r.get("MaturityDate", "")),
                   price=r.get("LastTradedPrice"),
                   yield_=r.get("Yield"))
        G.add_node(n_issuer,  kind="issuer",  name=issuer)
        G.add_node(n_sector,  kind="sector",  name=sector)
        G.add_node(n_rating,  kind="rating",  grade=rating)
        G.add_node(n_country, kind="country", name=country)
        G.add_node(n_btype,   kind="bondtype", name=btype)

        G.add_edge(n_bond,   n_issuer,  relation="issued_by")
        G.add_edge(n_bond,   n_rating,  relation="rated")
        G.add_edge(n_bond,   n_btype,   relation="instrument_of")
        G.add_edge(n_issuer, n_sector,  relation="operates_in")
        G.add_edge(n_issuer, n_country, relation="domiciled_in")

    KG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(KG_FILE, "wb") as f:
        pickle.dump(G, f)
    print(f"[kg] graph -> {KG_FILE.name}  ({G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges)")
    return G


def load_graph() -> nx.MultiDiGraph:
    global _graph
    if _graph is None:
        if not KG_FILE.exists():
            _graph = build_graph()
        else:
            with open(KG_FILE, "rb") as f:
                _graph = pickle.load(f)
    return _graph


def issuer_bonds(issuer: str) -> list[str]:
    G = load_graph()
    target = _node("issuer", issuer)
    return [u.split(":", 1)[1] for u, v, d in G.in_edges(target, data=True)
            if d.get("relation") == "issued_by"]


def issuer_exposure(issuer: str) -> dict:
    """Group exposure for an issuer: sector, country, ratings of its bonds."""
    G = load_graph()
    issuer_n = _node("issuer", issuer)
    if issuer_n not in G:
        return {}
    bonds = issuer_bonds(issuer)
    ratings = set()
    for b in bonds:
        for _, v, d in G.out_edges(_node("bond", b), data=True):
            if d.get("relation") == "rated":
                ratings.add(v.split(":", 1)[1])
    sectors, countries = set(), set()
    for _, v, d in G.out_edges(issuer_n, data=True):
        if d.get("relation") == "operates_in":
            sectors.add(v.split(":", 1)[1])
        if d.get("relation") == "domiciled_in":
            countries.add(v.split(":", 1)[1])
    return {
        "issuer":    issuer,
        "bond_count": len(bonds),
        "bonds":      bonds,
        "ratings":    sorted(ratings),
        "sectors":    sorted(sectors),
        "countries":  sorted(countries),
    }


def sector_issuers(sector: str) -> list[str]:
    G = load_graph()
    target = _node("sector", sector)
    return [u.split(":", 1)[1] for u, v, d in G.in_edges(target, data=True)
            if d.get("relation") == "operates_in"]


def bond_neighbours(isin: str) -> dict:
    G = load_graph()
    n = _node("bond", isin)
    if n not in G:
        return {}
    nbrs = {"issued_by": [], "rated": [], "instrument_of": []}
    for _, v, d in G.out_edges(n, data=True):
        rel = d.get("relation")
        if rel in nbrs:
            nbrs[rel].append(v.split(":", 1)[1])
    return nbrs


if __name__ == "__main__":
    build_graph()
