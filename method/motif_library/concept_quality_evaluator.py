import argparse
import json
import random
import re
from collections import Counter


def _is_factor(node: dict) -> bool:
    return node.get("type") != "stance"


def _tokens(text: str):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _normalize(text: str) -> str:
    t = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def _jaccard(a: set, b: set) -> float:
    return (len(a & b) / len(a | b)) if (a or b) else 0.0


def _sample_pairs(n: int, k: int):
    if n < 2 or k <= 0:
        return []
    random.seed(0)
    pairs = set()
    while len(pairs) < k:
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pairs.add((i, j))
    return list(pairs)


def evaluate(graph: dict) -> dict:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    factors = [n for n in nodes if _is_factor(n)]

    names = [str(n.get("id", "")) for n in factors]
    toks = [set(_tokens(x)) for x in names]
    norms = [_normalize(x) for x in names]
    total = len(names)
    unique = len(set(norms))
    avg_tokens = (sum(len(t) for t in toks) / total) if total else 0.0

    factor_ids = set(names)
    used = set()
    contributes_from = set()
    causal_edges = 0
    for e in edges:
        s, t, r = e.get("source"), e.get("target"), e.get("relation")
        if s in factor_ids:
            used.add(s)
        if t in factor_ids:
            used.add(t)
        if r == "contributes" and s in factor_ids:
            contributes_from.add(s)
        if r == "causal":
            causal_edges += 1

    isolated = total - len(used)
    contributes_cov = (len(contributes_from) / total) if total else 0.0
    causal_density = causal_edges / (max(1, total * (total - 1)))

    source_sets = [set(n.get("sources", []) or []) for n in factors]
    src_counts = Counter(x for ss in source_sets for x in ss)
    unique_sources = len(src_counts)
    total_assignments = sum(src_counts.values())
    concepts_per_source = (total_assignments / unique_sources) if unique_sources else 0.0

    pair_cap = min(10000, (total * (total - 1)) // 2)
    pairs = _sample_pairs(total, pair_cap)
    if pairs:
        lex_j = sum(_jaccard(toks[i], toks[j]) for i, j in pairs) / len(pairs)
        src_j = sum(_jaccard(source_sets[i], source_sets[j]) for i, j in pairs) / len(pairs)
        src_any = sum(1 for i, j in pairs if source_sets[i] & source_sets[j]) / len(pairs)
    else:
        lex_j = src_j = src_any = 0.0

    return {
        "counts": {
            "factors": total,
            "unique_normalized_factors": unique,
            "duplication_rate": (1 - unique / total) if total else 0.0,
            "unique_sources": unique_sources,
        },
        "length": {"avg_tokens_per_factor": avg_tokens},
        "graph": {
            "isolated_factor_ratio": (isolated / total) if total else 0.0,
            "contributes_coverage": contributes_cov,
            "causal_edges": causal_edges,
            "causal_density_estimate": causal_density,
        },
        "source_overlap": {
            "avg_jaccard_across_concepts": src_j,
            "any_overlap_rate": src_any,
            "avg_concepts_per_source": concepts_per_source,
        },
        "lexical_redundancy": {"avg_name_token_jaccard": lex_j},
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate a concept list's quality metrics")
    ap.add_argument("--graph", required=True, help="Path to participant_context_graph.json")
    ap.add_argument("--out", default=None, help="Optional output metrics JSON path")
    args = ap.parse_args()

    with open(args.graph, "r", encoding="utf-8") as f:
        graph = json.load(f)
    metrics = evaluate(graph)

    out_path = args.out or re.sub(r"\.json$", "", args.graph) + ".quality.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
