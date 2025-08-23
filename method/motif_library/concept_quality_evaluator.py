import os
import re
import json
import logging
from typing import List, Dict, Tuple
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_GB_US_MAP = {
    "colour": "color",
    "favourite": "favorite",
    "behaviour": "behavior",
    "organise": "organize",
    "organisation": "organization",
    "centre": "center",
    "theatre": "theater",
    "metre": "meter",
}

_ABBREV_MAP = {
    "w/": "with",
    "w/o": "without",
    "btw": "by the way",
    "i.e.": "that is",
    "e.g.": "for example",
    "info": "information",
}

_DET_PATTERN = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9\s\-_/]+")
_MULTISPACE = re.compile(r"\s+")


def normalize_concept(text: str) -> str:
    t = text.strip()
    t = t.lower()
    t = _DET_PATTERN.sub("", t)
    for k, v in _ABBREV_MAP.items():
        t = t.replace(k, v)
    words = t.split()
    norm_words: List[str] = []
    for w in words:
        w2 = _GB_US_MAP.get(w, w)
        # naive lemmatization: plural -> singular
        if len(w2) > 3 and w2.endswith("s") and not w2.endswith("ss"):
            w2 = w2[:-1]
        norm_words.append(w2)
    t = " ".join(norm_words)
    t = _NON_ALNUM.sub(" ", t)
    t = _MULTISPACE.sub(" ", t).strip()
    return t


def load_concepts_from_graph(graph_file: str) -> List[str]:
    with open(graph_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [n["id"] for n in data.get("nodes", []) if n.get("type") != "stance"]


def embed_texts(model: SentenceTransformer, texts: List[str]):
    return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)


def cluster_by_similarity(labels: List[str], embeddings, threshold: float) -> List[List[int]]:
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

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def choose_preferred_label(indices: List[int], labels: List[str], embeddings) -> Tuple[str, List[Tuple[str, float]]]:
    vecs = embeddings[indices]
    centroid = vecs.mean(dim=0, keepdim=True)
    sims = util.pytorch_cos_sim(vecs, centroid).squeeze(1)
    scored = [(labels[idx], sims[k].item()) for k, idx in enumerate(indices)]
    scored.sort(key=lambda x: (-x[1], len(x[0])))
    return scored[0][0], scored


def derive_relations(members: List[str]) -> Dict[str, List[Dict[str, str]]]:
    tokens = [m.split() for m in members]
    broader: List[Dict[str, str]] = []
    instance_of: List[Dict[str, str]] = []
    for i, a in enumerate(members):
        for j, b in enumerate(members):
            if i == j:
                continue
            ta, tb = set(tokens[i]), set(tokens[j])
            jacc = len(ta & tb) / max(1, len(ta | tb))
            if len(ta) < len(tb) and (a in b or jacc >= 0.6):
                broader.append({"broader": a, "narrower": b})
            if any(ch.isdigit() for ch in b):
                instance_of.append({"instance": b, "class": a})
    # de-dup
    seen = set()
    b2 = []
    for rel in broader:
        key = (rel["broader"], rel["narrower"])
        if key not in seen:
            seen.add(key)
            b2.append(rel)
    seen = set()
    i2 = []
    for rel in instance_of:
        key = (rel["instance"], rel["class"])
        if key not in seen:
            seen.add(key)
            i2.append(rel)
    return {"broader_narrower": b2, "instance_of": i2}


def merge_concepts(graph_file: str, threshold: float = 0.72, model_name: str = "all-MiniLM-L6-v2") -> None:
    logger.info(f"Loading concepts from {graph_file}")
    originals = load_concepts_from_graph(graph_file)
    norm_map: Dict[str, str] = {c: normalize_concept(c) for c in originals}
    normals = list({v for v in norm_map.values() if v})
    if not normals:
        logger.warning("No concepts to merge.")
        return

    logger.info(f"Embedding {len(normals)} concepts with {model_name}")
    model = SentenceTransformer(model_name)
    emb = embed_texts(model, normals)

    logger.info(f"Clustering with threshold={threshold}")
    clusters = cluster_by_similarity(normals, emb, threshold)

    cluster_outputs = []
    label_map: Dict[str, str] = {}
    for comp in clusters:
        members = [normals[i] for i in comp]
        preferred, scored = choose_preferred_label(comp, normals, emb)
        rels = derive_relations(members)
        # map every normalized and its originals to preferred
        for m in members:
            label_map[m] = preferred
        cluster_outputs.append({
            "preferred_label": preferred,
            "members": members,
            "scores": scored,
            "relations": rels,
        })

    # Back-map originals -> preferred
    original_to_preferred = {orig: label_map.get(nrm, nrm) for orig, nrm in norm_map.items()}

    out_dir = os.path.dirname(graph_file)
    out_path = os.path.join(out_dir, "concept_merge_results.json")
    result = {
        "graph_file": graph_file,
        "threshold": threshold,
        "model": model_name,
        "normalization_map": norm_map,
        "clusters": cluster_outputs,
        "original_to_preferred": original_to_preferred,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved merge results to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Normalize, merge, and level concept list by semantic similarity")
    parser.add_argument("--graph", required=True, help="Path to participant_context_graph.json")
    parser.add_argument("--threshold", type=float, default=0.72, help="Similarity threshold for merging")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    args = parser.parse_args()
    merge_concepts(args.graph, threshold=args.threshold, model_name=args.model)
