import argparse
import json
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util


def _is_factor(node: dict) -> bool:
    return node.get("type") != "stance"


def _normalize(text: str) -> str:
    t = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def _embed_texts(model: SentenceTransformer, texts):
    return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)


def _cluster_by_similarity(labels, embeddings, threshold: float):
    sims = util.pytorch_cos_sim(embeddings, embeddings)
    n = len(labels)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j].item() >= threshold:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def _abstract_label(indices, raw_names, token_sets):
    # tokens that appear in >=60% of members
    from math import ceil
    need = max(1, ceil(0.6 * len(indices)))
    freq = defaultdict(int)
    for idx in indices:
        for tok in token_sets[idx]:
            freq[tok] += 1
    common = [t for t, c in freq.items() if c >= need]
    if common:
        common.sort(key=lambda x: (len(x), x))
        # choose up to 3 tokens for a compact abstract label
        return " ".join(common[:3])
    # fallback: pick the member closest to centroid and shorter label
    # compute centroid in embedding space later in merge (we'll just pick shortest for simplicity here)
    return sorted([raw_names[i] for i in indices], key=lambda s: (len(s), s))[0]


def merge_graph(graph: dict, threshold: float = 0.72, model_name: str = "all-MiniLM-L6-v2") -> dict:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    stance_nodes = [n for n in nodes if not _is_factor(n)]
    factor_nodes = [n for n in nodes if _is_factor(n)]

    if not factor_nodes:
        return {"nodes": nodes, "edges": edges}

    raw_names = [str(n.get("id", "")) for n in factor_nodes]
    norm_names = [_normalize(x) for x in raw_names]
    token_sets = [set(norm.split()) for norm in norm_names]

    model = SentenceTransformer(model_name)
    emb = _embed_texts(model, norm_names)
    clusters = _cluster_by_similarity(norm_names, emb, threshold)

    id_map = {}
    merged_nodes = []
    for comp in clusters:
        label = _abstract_label(comp, raw_names, token_sets)
        # map all originals to this abstract label
        for idx in comp:
            id_map[raw_names[idx]] = label
        # aggregate counts and sources
        total_count = sum(int(factor_nodes[idx].get("count", 1)) for idx in comp)
        src = set()
        for idx in comp:
            src.update(factor_nodes[idx].get("sources", []) or [])
        # preserve originals and evidences
        original_ids = sorted({raw_names[idx] for idx in comp})
        originals = []
        evidences = []
        seen_ev = set()
        for idx in comp:
            node = factor_nodes[idx]
            item = {
                "id": raw_names[idx],
                "count": int(node.get("count", 1)),
                "sources": node.get("sources", []),
            }
            ev = node.get("evidence")
            if ev is not None:
                item["evidence"] = ev
                if isinstance(ev, list):
                    for e in ev:
                        if e and e not in seen_ev:
                            seen_ev.add(e)
                            evidences.append(e)
                else:
                    if ev and ev not in seen_ev:
                        seen_ev.add(ev)
                        evidences.append(ev)
            originals.append(item)
        merged_nodes.append({
            "id": label,
            "type": "factor",
            "count": total_count,
            "sources": sorted(src),
            "original_ids": original_ids,
            "originals": originals,
            "evidences": evidences,
        })

    # deduplicate merged_nodes by id
    dedup = {}
    for n in merged_nodes:
        dedup[n["id"]] = n
    merged_nodes = list(dedup.values())

    # remap edges and combine duplicates
    edge_map = {}
    for e in edges:
        s = id_map.get(e.get("source"), e.get("source"))
        t = id_map.get(e.get("target"), e.get("target"))
        rel = e.get("relation")
        cue = e.get("cue")
        if rel == "causal" and s == t:
            continue  # drop self loops created by merge
        key = (s, t, rel, cue)
        if key not in edge_map:
            edge_map[key] = {
                "source": s,
                "target": t,
                "relation": rel,
            }
            if cue is not None:
                edge_map[key]["cue"] = cue
            edge_map[key]["count"] = int(e.get("count", 1))
        else:
            edge_map[key]["count"] += int(e.get("count", 1))

    merged_edges = list(edge_map.values())

    return {"nodes": stance_nodes + merged_nodes, "edges": merged_edges}


def main():
    ap = argparse.ArgumentParser(description="Semantic-merge concepts with embeddings and abstract labels")
    ap.add_argument("--graph", required=True, help="Path to participant_context_graph.json")
    ap.add_argument("--out", default=None, help="Output path for merged graph JSON")
    ap.add_argument("--threshold", type=float, default=0.72, help="Cosine similarity threshold for clustering")
    ap.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    args = ap.parse_args()

    with open(args.graph, "r", encoding="utf-8") as f:
        graph = json.load(f)
    merged = merge_graph(graph, threshold=args.threshold, model_name=args.model)
    out_path = args.out or re.sub(r"\.json$", "", args.graph) + ".merged.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(out_path)


if __name__ == "__main__":
    main()


