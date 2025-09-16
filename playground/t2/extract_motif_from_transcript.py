"""Extract small reasoning motifs from belief-update transcripts."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple, Iterable
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent.parent

import sys

sys.path.insert(0, str(ROOT_DIR))
from playground.t2.llm_utils import QwenLLM

import dotenv

dotenv.load_dotenv(ROOT_DIR / ".env")

# Define motif type descriptions
MOTIF_TYPES = {
    "M1": "Chain",
    "M2.1": "Basic Fork (1-to-2)",
    "M2.2": "Extended Fork (1-to-3)",
    "M2.3": "Large Fork (1-to-4+)",
    "M3.1": "Basic Collider (2-to-1)",
    "M3.2": "Extended Collider (3-to-1)",
    "M3.3": "Large Collider (4+-to-1)",
}
# TODO: currently only extracting 3-node motifs


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def escape_mermaid_label(label: str) -> str:
    return str(label).replace('"', "'")


def _build_llm_prompt(record: Dict[str, object]) -> str:
    prolific_id: str = str(record.get("prolific_id", "unknown"))
    topic: str = str(record.get("topic", "")) or "healthcare"
    demographics = record.get("demographics", {}) or {}
    qas: List[Dict[str, object]] = record.get("context_qas", []) or []

    qas_lines: List[str] = []
    for qa in qas:
        qnum = str(qa.get("question_number", ""))
        question = str(qa.get("question", "")).strip()
        answer = str(qa.get("answer", "")).strip()
        qas_lines.append(f"Q{qnum} Question: {question}\nQ{qnum} Answer: {answer}")

    qas_block = "\n\n".join(qas_lines)

    schema_block = (
        "Return ONLY a JSON array of motif objects. Each motif must have exactly these fields: \n"
        '- nodes: ["n1", "n2", "n3"] (IDs only)\n'
        '- edges: exactly two directed edges, e.g., [["n1", "n2"], ["n2", "n3"]]\n'
        '- node_labels: {"n1": <neutral value/factor>, "n2": <neutral value/factor>, "n3": <neutral value/factor>}\n'
        "- demographics: copy of participant demographics\n"
        '- sources: array of strings referencing supporting QA ids in the form "%s_q<question_number>"\n'
        % prolific_id
    )

    aspect_block = ""

    allowed_types_block = (
        "Allowed motif types (use EXACTLY 2 edges):\n"
        "- M1 Chain: A -> B, B -> C\n"
        "- M2.1 Basic Fork (1-to-2): A -> B, A -> C\n"
        "- M3.1 Basic Collider (2-to-1): A -> C, B -> C\n"
        "All node labels must be neutral values/factors. Do NOT include words like 'effect', 'positive', 'negative', 'strong', 'weak', 'immediate', 'gradual', 'support', 'oppose'.\n"
    )

    scoring_block = (
        "Also include two OPTIONAL keys when you can infer them from the QAs: \n"
        "- node_values: {<label>: {\"valence\": one of ['pro','con','neutral'], \"confidence\": 0..1}}\n"
        '- edge_strengths: {["src_label","dst_label"]: {"strength": 0..1, "confidence": 0..1}}\n'
        "Use best-effort estimates; omit if unclear.\n"
    )

    instruction = (
        "Extract the participant's small reasoning motifs as 3-node graphs. "
        "Nodes must be neutral; do not include opinion/effect words in node labels. "
        "Use ONLY the allowed motif types and their edge patterns. Use exactly 2 edges. "
        "Keep the exact motif object schema. Return ONLY the JSON array."
    )

    prompt = (
        f"Participant prolific_id: {prolific_id}\n"
        f"Topic: {topic}\n\n"
        f"Demographics (verbatim JSON):\n{json.dumps(demographics, ensure_ascii=False)}\n\n"
        f"Transcript QAs:\n{qas_block}\n\n"
        f"{schema_block}\n\n{allowed_types_block}{scoring_block}{instruction}"
    )
    return prompt


def _validate_and_fix_motifs(
    motifs: List[Dict[str, object]], prolific_id: str, demographics: Dict[str, object]
) -> List[Dict[str, object]]:
    fixed: List[Dict[str, object]] = []
    # Allowed two-edge patterns for 3-node motifs
    allowed_edge_sets = {
        ("n1->n2", "n2->n3"),  # M1 Chain
        ("n1->n2", "n1->n3"),  # M2.1 Fork
        ("n1->n3", "n2->n3"),  # M3.1 Collider
    }
    for m in motifs or []:
        try:
            nodes = ["n1", "n2", "n3"]
            node_labels = m.get("node_labels", {}) or {}
            # Require all labels present and neutral
            if (
                not node_labels.get("n1")
                or not node_labels.get("n2")
                or not node_labels.get("n3")
            ):
                continue
            for key in ("n1", "n2", "n3"):
                label = str(node_labels.get(key, "")).lower()
                if any(
                    bad in label
                    for bad in [
                        "effect",
                        "positive",
                        "negative",
                        "strong",
                        "weak",
                        "immediate",
                        "gradual",
                        "support",
                        "oppose",
                    ]
                ):
                    # Reject if label contains opinion/effect words
                    continue

            # Validate exactly two edges and allowed pattern
            edges_in = m.get("edges", []) or []
            if not isinstance(edges_in, list) or len(edges_in) != 2:
                continue

            def fmt_edge(e: List[str]) -> str:
                return f"{e[0]}->{e[1]}" if isinstance(e, list) and len(e) == 2 else ""

            edge_keys = tuple(sorted(fmt_edge(e) for e in edges_in))
            if edge_keys not in allowed_edge_sets:
                continue

            edges = [[str(a), str(b)] for a, b in edges_in]
            sources = m.get("sources") or [prolific_id]
            if isinstance(sources, str):
                sources = [sources]

            motif_out: Dict[str, object] = {
                "nodes": nodes,
                "edges": edges,
                "node_labels": {
                    "n1": str(node_labels["n1"]).strip(),
                    "n2": str(node_labels["n2"]).strip(),
                    "n3": str(node_labels["n3"]).strip(),
                },
                "demographics": demographics or {},
                "sources": [str(s) for s in sources if s],
            }
            # Optional scoring passthrough if present
            if isinstance(m.get("node_values"), dict):
                motif_out["node_values"] = m["node_values"]
            if isinstance(m.get("edge_strengths"), dict):
                motif_out["edge_strengths"] = m["edge_strengths"]

            fixed.append(motif_out)
        except Exception:
            continue
    return fixed


def _extract_motifs_via_llm(record: Dict[str, object]) -> List[Dict[str, object]]:
    prolific_id: str = str(record.get("prolific_id", "unknown"))
    demographics: Dict[str, object] = record.get("demographics", {}) or {}
    try:
        llm = QwenLLM(model="qwen-plus")
    except Exception:
        return []

    system_message = "You extract small graphs of reasoning motifs from transcripts and output strict JSON only."
    prompt = _build_llm_prompt(record)
    response = llm.generate_response(
        prompt,
        system_message=system_message,
        temperature=0.1,
        return_json=True,
        debug=False,
        max_retries=2,
        max_tokens=8192,
    )

    motifs_json: Optional[List[Dict[str, object]]] = None
    if isinstance(response, list):
        motifs_json = response
    elif isinstance(response, dict) and "motifs" in response:
        motifs_json = response.get("motifs")  # type: ignore[assignment]
    else:
        motifs_json = None

    if not motifs_json:
        return []
    return _validate_and_fix_motifs(motifs_json, prolific_id, demographics)


def extract_motifs_from_record(
    record: Dict[str, object],
) -> Tuple[str, Dict[str, object]]:
    prolific_id: str = str(record.get("prolific_id", "unknown"))
    topic: str = str(record.get("topic", "")) or "healthcare"

    # LLM-only extraction
    motifs = _extract_motifs_via_llm(record)
    motifs = _deduplicate_motifs(motifs)

    return prolific_id, {"prolific_id": prolific_id, "topic": topic, "motifs": motifs}


def _deduplicate_motifs(motifs: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Merge motifs that are identical up to label/edge equality.

    Signature = (sorted tuple of labels), (sorted tuple of directed label->label edges).
    For duplicates, union the 'sources' list and keep the higher-confidence
    estimates in 'node_values' and 'edge_strengths' when both are present.
    """
    by_signature: Dict[
        Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]], Dict[str, object]
    ] = {}

    def _labels_by_id(m: Dict[str, object]) -> Dict[str, str]:
        node_labels = m.get("node_labels", {}) or {}
        return {k: str(v) for k, v in node_labels.items()}

    def _edge_pairs_by_label(
        m: Dict[str, object], id2label: Dict[str, str]
    ) -> Tuple[Tuple[str, str], ...]:
        edges = m.get("edges", []) or []
        pairs: List[Tuple[str, str]] = []
        for e in edges:
            if isinstance(e, list) and len(e) == 2:
                a, b = str(e[0]), str(e[1])
                pairs.append((id2label.get(a, a), id2label.get(b, b)))
        return tuple(sorted(pairs))

    def _merge_sources(a: object, b: object) -> List[str]:
        a_list = a if isinstance(a, list) else []
        b_list = b if isinstance(b, list) else []
        return sorted({str(x) for x in list(a_list) + list(b_list) if x})

    for m in motifs or []:
        try:
            id2label = _labels_by_id(m)
            labels_sorted = tuple(sorted(id2label.values()))
            edges_by_label = _edge_pairs_by_label(m, id2label)
            signature = (labels_sorted, edges_by_label)

            if signature not in by_signature:
                by_signature[signature] = m
            else:
                existing = by_signature[signature]
                # Merge sources
                existing["sources"] = _merge_sources(
                    existing.get("sources"), m.get("sources")
                )
        except Exception:
            # If anything unexpected, keep original motif
            by_signature[(tuple(), tuple())] = m

    return list(by_signature.values())


def read_jsonl(
    path: str, max_records: Optional[int] = None
) -> Iterable[Dict[str, object]]:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield obj
                count += 1
                if max_records is not None and count >= max_records:
                    return
            except json.JSONDecodeError:
                continue


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def motifs_to_mermaid(motifs: List[Dict[str, object]]) -> str:
    lines: List[str] = ["flowchart LR"]
    for idx, m in enumerate(motifs):
        lbl = m.get("node_labels", {})
        # Use label-based IDs; ensure uniqueness per motif via idx
        n1_label = escape_mermaid_label(lbl.get("n1", f"node1_{idx}"))
        n2_label = escape_mermaid_label(lbl.get("n2", f"node2_{idx}"))
        n3_label = escape_mermaid_label(lbl.get("n3", f"node3_{idx}"))
        id1 = f"A{idx}"
        id2 = f"B{idx}"
        id3 = f"C{idx}"
        lines.append(f'{id1}["{n1_label}"]')
        lines.append(f'{id2}["{n2_label}"]')
        lines.append(f'{id3}["{n3_label}"]')
        # Draw edges from the motif's validated edges
        for a, b in m.get("edges", []) or []:
            src = id1 if a == "n1" else id2 if a == "n2" else id3
            dst = id1 if b == "n1" else id2 if b == "n2" else id3
            # If edge_strengths provided, annotate edge label with strength
            edge_label = ""
            strengths = (
                m.get("edge_strengths")
                if isinstance(m.get("edge_strengths"), dict)
                else None
            )
            if strengths:
                key = f"[{lbl.get(a, '')},{lbl.get(b, '')}]"
                val = (
                    strengths.get(key) if isinstance(strengths.get(key), dict) else None
                )
                if val and isinstance(val.get("strength"), (int, float)):
                    edge_label = f" | {float(val['strength']):.2f}"
            lines.append(
                f"{src} -->|{edge_label}| {dst}" if edge_label else f"{src} --> {dst}"
            )
    return "\n".join(lines)


def write_per_prolific(
    output_dir: str, prolific_id: str, data: Dict[str, object]
) -> Dict[str, str]:
    # Create per-participant folder
    folder = os.path.join(output_dir, prolific_id)
    ensure_dir(folder)

    # Write JSON
    json_path = os.path.join(folder, f"{prolific_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Write Mermaid
    mermaid_str = motifs_to_mermaid(data.get("motifs", []) or [])
    mmd_path = os.path.join(folder, f"{prolific_id}.mmd")
    with open(mmd_path, "w", encoding="utf-8") as f:
        f.write(mermaid_str + "\n")

    # Write Markdown wrapper
    md_path = os.path.join(folder, f"{prolific_id}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Motifs\n\n")
        f.write("```mermaid\n")
        f.write(mermaid_str + "\n")
        f.write("```\n")

    return {"json": json_path, "mmd": mmd_path, "md": md_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract small reasoning motifs from transcripts"
    )
    parser.add_argument(
        "--input",
        default="data/sample_belief_update_healthcare.jsonl",
        help="Path to JSONL input",
    )
    parser.add_argument(
        "--output_dir",
        default="playground/t2/results/motifs_from_transcripts",
        help="Directory to write per-prolific outputs",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Limit number of records to process",
    )
    args = parser.parse_args()

    # topic = args.input.split("/")[-1].split("_")[-1].split(".")[0]
    # print(f"Topic: {topic}")
    # support_label = f"Support for {topic}"
    # print(f"Support label: {support_label}")

    # Count total records first
    print("Counting total records...")
    total_records = sum(1 for _ in read_jsonl(str(ROOT_DIR / args.input), args.max_records))
    
    written = 0
    seen_ids = set()
    
    with tqdm(total=total_records, desc="Processing", position=0, leave=True) as pbar:
        for record in read_jsonl(str(ROOT_DIR / args.input)):
            prolific_id = str(record.get("prolific_id", "unknown"))

            if prolific_id in seen_ids:
                pbar.update(1)
                continue
            
            pbar.set_postfix({"ID": prolific_id[:8]})
            _, out_data = extract_motifs_from_record(record)
            seen_ids.add(prolific_id)
            paths = write_per_prolific(
                str(ROOT_DIR / args.output_dir), prolific_id, out_data
            )

            written += 1
            pbar.update(1)
            
            if args.max_records is not None and written >= args.max_records:
                break

    print(f"Wrote {written} participant folders to {str(ROOT_DIR / args.output_dir)}")


if __name__ == "__main__":
    main()
