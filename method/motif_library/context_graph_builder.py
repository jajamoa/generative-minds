"""
Context Graph Builder - Extracts contributing factors and simple causal traces
from Reddit comments and builds per-comment and aggregated context graphs.

For each participant's train split ("<participant>-train.json") in
data/30ppl_eval_v1_0728/<participant>/, this module constructs:
  - Per-comment context graph linking extracted factors/aspects to the stance
  - An aggregated participant-level graph merged over all train comments

Outputs are saved under intermediate_data/30ppl_eval_v1_0728/<participant>/
with a folder structure mirroring data/.
"""

import json
import os
import re
import time
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import networkx as nx
import requests
from dotenv import load_dotenv

import sys

# Ensure parent (method/) is importable when running this file directly
_THIS_DIR = Path(__file__).resolve().parent
_METHOD_ROOT = _THIS_DIR.parent
if str(_METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(_METHOD_ROOT))

from utils.llm import (
    BaseContextLLM,
    ContextLLMConfig,
    QwenMaxContextLLM,
)
from data.data_preprocess import DataPreprocessor


import yaml
from utils.logging import setup_logging
from tqdm import tqdm

# Load .env then set up logging
load_dotenv(dotenv_path=_METHOD_ROOT / ".env")
setup_logging()
logger = logging.getLogger(__name__)


STOPWORDS = set(
    [
        "the",
        "and",
        "or",
        "if",
        "but",
        "when",
        "what",
        "which",
        "this",
        "that",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "they",
        "their",
        "it",
        "its",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "for",
        "in",
        "on",
        "at",
        "as",
        "with",
        "by",
        "from",
        "about",
        "into",
        "over",
        "after",
        "before",
        "so",
        "because",
        "due",
        "since",
        "therefore",
        "thus",
        "then",
        "also",
        "too",
        "very",
        "not",
        "no",
        "do",
        "did",
        "does",
        "have",
        "has",
        "had",
        "a",
        "an",
    ]
)


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_phrase(phrase: str) -> str:
    phrase = re.sub(r"\s+", " ", phrase)
    phrase = re.sub(r"^[\s,;.:-]+|[\s,;.:-]+$", "", phrase)
    return phrase.strip()


def _sentence_split(text: str) -> List[str]:
    # Simple sentence splitter
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in sentences if s and s.strip()]


def _extract_causal_pairs(sentence: str) -> List[Tuple[str, str, str]]:
    """
    Extract simple causal pairs from a sentence using surface patterns.
    Returns list of (cause, effect, cue).
    """
    pairs: List[Tuple[str, str, str]] = []
    s = " " + sentence + " "
    s_norm = s.lower()

    # Pattern: effect because cause
    if " because " in s_norm:
        m = re.search(r"(.+?)\bbecause\b(.+)", sentence, flags=re.IGNORECASE)
        if m:
            effect = _clean_phrase(m.group(1))
            cause = _clean_phrase(m.group(2))
            if effect and cause:
                pairs.append((cause, effect, "because"))

    # Pattern: effect due to cause
    if " due to " in s_norm:
        m = re.search(r"(.+?)\bdue to\b(.+)", sentence, flags=re.IGNORECASE)
        if m:
            effect = _clean_phrase(m.group(1))
            cause = _clean_phrase(m.group(2))
            if effect and cause:
                pairs.append((cause, effect, "due to"))

    # Pattern: cause leads to effect
    if " leads to " in s_norm:
        m = re.search(r"(.+?)\bleads to\b(.+)", sentence, flags=re.IGNORECASE)
        if m:
            cause = _clean_phrase(m.group(1))
            effect = _clean_phrase(m.group(2))
            if effect and cause:
                pairs.append((cause, effect, "leads to"))

    # Pattern: cause so effect
    if re.search(r"\bso\b", s_norm):
        m = re.search(r"(.+?)\bso\b(.+)", sentence, flags=re.IGNORECASE)
        if m:
            cause = _clean_phrase(m.group(1))
            effect = _clean_phrase(m.group(2))
            if effect and cause:
                pairs.append((cause, effect, "so"))

    # Pattern: cause therefore effect
    if " therefore " in s_norm:
        m = re.search(r"(.+?)\btherefore\b(.+)", sentence, flags=re.IGNORECASE)
        if m:
            cause = _clean_phrase(m.group(1))
            effect = _clean_phrase(m.group(2))
            if effect and cause:
                pairs.append((cause, effect, "therefore"))

    return pairs


def _extract_keyword_factors(text: str, max_keywords: int = 8) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-'_]{2,}", text.lower())
    freq: Dict[str, int] = defaultdict(int)
    for w in words:
        if w in STOPWORDS:
            continue
        freq[w] += 1

    # Sort by frequency then alphabetically for stability
    sorted_words = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    keywords = [kw for kw, _ in sorted_words[:max_keywords]]
    return keywords


class ContextGraphBuilder:
    """
    Build per-comment and aggregated context graphs from participant train data.
    """

    def __init__(
        self,
        dataset_dir: str = "data/30ppl_eval_v1_0728",
        output_root: str = "intermediate_data/30ppl_eval_v1_0728",
        llm: Optional["BaseContextLLM"] = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.llm = llm

    def _build_comment_graph(
        self,
        comment_text: str,
        stance_label: str,
        post_title: Optional[str] = None,
        scenario: Optional[str] = None,
        debug_dir: Optional[Path] = None,
    ) -> nx.DiGraph:
        """Construct a context graph for one comment."""
        G = nx.DiGraph()
        text = _normalize_text(comment_text)
        all_factors: List[str]
        causal_pairs: List[Tuple[str, str, str]]

        # Prefer LLM extraction when available
        if self.llm is not None:
            try:
                ctx = self.llm.generate_context(
                    post_title=post_title or "",
                    scenario=scenario or "",
                    comment=text,
                    stance=stance_label or "",
                )
                # Expected keys: {"factors": List[str], "causal_pairs": [{cause,effect,cue?}], "_raw"?: str}
                factors = ctx.get("factors", [])
                pairs_in = ctx.get("causal_pairs", [])
                causal_pairs = []
                for p in pairs_in:
                    cause = _clean_phrase(str(p.get("cause", "")))
                    effect = _clean_phrase(str(p.get("effect", "")))
                    cue = (
                        _clean_phrase(str(p.get("cue", "")))
                        if p.get("cue")
                        else "causal"
                    )
                    if cause and effect:
                        causal_pairs.append((cause, effect, cue))
                # Merge factors from pairs too
                factors_from_pairs = [c for c, _, _ in causal_pairs] + [
                    e for _, e, _ in causal_pairs
                ]
                keyword_factors = [str(f) for f in factors]
                combined = factors_from_pairs + keyword_factors
                all_factors = []
                seen = set()
                for f in combined:
                    f_norm = _clean_phrase(f)
                    if not f_norm:
                        continue
                    key = f_norm.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    all_factors.append(f_norm)
                # Save raw for debugging if requested
                if debug_dir is not None and "_raw" in ctx:
                    try:
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        with open(
                            debug_dir / "llm_response.json", "w", encoding="utf-8"
                        ) as df:
                            df.write(ctx["_raw"])  # raw string content
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(
                    f"LLM extraction failed, falling back to heuristics: {e}"
                )
                # fall back to heuristics below
                self.llm = None

        if self.llm is None:
            sentences = _sentence_split(text)
            # Extract causal pairs heuristically
            causal_pairs = []
            factors_from_pairs = []
            for sent in sentences:
                pairs = _extract_causal_pairs(sent)
                for cause, effect, cue in pairs:
                    causal_pairs.append((cause, effect, cue))
                    factors_from_pairs.extend([cause, effect])
            # Extract keyword factors
            keyword_factors = _extract_keyword_factors(text)
            # Deduplicate and normalize factors
            all_factors = []
            seen = set()
            for f in factors_from_pairs + keyword_factors:
                f_norm = _clean_phrase(f)
                if not f_norm:
                    continue
                if f_norm.lower() in seen:
                    continue
                seen.add(f_norm.lower())
                all_factors.append(f_norm)

        # Add nodes
        stance_node = f"stance::{stance_label}"
        G.add_node(stance_node, type="stance", label=stance_label)

        for factor in all_factors:
            G.add_node(factor, type="factor")
            # Connect factor to stance as contributing
            G.add_edge(
                factor,
                stance_node,
                relation="contributes",
            )

        # Add causal edges among factors
        for cause, effect, cue in causal_pairs:
            cause_n = _clean_phrase(cause)
            effect_n = _clean_phrase(effect)
            if not cause_n or not effect_n:
                continue
            # Ensure nodes exist
            if not G.has_node(cause_n):
                G.add_node(cause_n, type="factor")
            if not G.has_node(effect_n):
                G.add_node(effect_n, type="factor")
            G.add_edge(cause_n, effect_n, relation="causal", cue=cue)

        return G

    def _generate_mermaid(self, G: nx.DiGraph, title: str) -> str:
        mermaid: List[str] = ["graph TD;"]
        node_ids: Dict[str, str] = {}

        for i, node in enumerate(G.nodes()):
            node_id = f"n{i}"
            node_ids[node] = node_id
            ntype = G.nodes[node].get("type", "factor")
            if ntype == "stance":
                mermaid.append(f'    {node_id}["{node}"]:::stanceNode;')
            else:
                mermaid.append(f'    {node_id}["{node}"]:::factorNode;')

        for u, v, data in G.edges(data=True):
            u_id = node_ids[u]
            v_id = node_ids[v]
            relation = data.get("relation", "")
            if relation == "causal":
                cue = data.get("cue", "")
                label = cue if cue else "causal"
                mermaid.append(f"    {u_id} -->|{label}| {v_id};")
            elif relation == "contributes":
                mermaid.append(f"    {u_id} -.->|contributes| {v_id};")
            else:
                mermaid.append(f"    {u_id} --> {v_id};")

        mermaid.append(
            "    classDef stanceNode fill:#f9f,stroke:#333,stroke-width:2px;"
        )
        mermaid.append(
            "    classDef factorNode fill:#bbf,stroke:#333,stroke-width:1px;"
        )
        mermaid.append(f'    title["{title}"]:::title;')
        mermaid.append("    classDef title fill:none,stroke:none;")
        return "\n".join(mermaid)

    def _save_graph_as_json(self, G: nx.DiGraph, file_path: Path) -> None:
        graph_data: Dict[str, Any] = {"nodes": [], "edges": []}
        for node, attrs in G.nodes(data=True):
            graph_data["nodes"].append({"id": node, **attrs})
        for u, v, attrs in G.edges(data=True):
            graph_data["edges"].append({"source": u, "target": v, **attrs})
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

    def _merge_into_aggregated(self, base: nx.DiGraph, addition: nx.DiGraph) -> None:
        for node, attrs in addition.nodes(data=True):
            if not base.has_node(node):
                base.add_node(node, **attrs)
        for u, v, attrs in addition.edges(data=True):
            if base.has_edge(u, v):
                # Maintain a simple count for aggregated edges
                base[u][v]["count"] = base[u][v].get("count", 1) + 1
            else:
                base.add_edge(u, v, **attrs)
                base[u][v]["count"] = 1

    def process_participant(self, participant: str) -> Dict[str, Any]:
        """
        Build context graphs for all train comments of a participant.
        """
        try:
            participant_dir = self.dataset_dir / participant
            train_file = participant_dir / f"{participant}-train.json"
            if not train_file.exists():
                return {
                    "participant": participant,
                    "error": f"Missing train file: {train_file}",
                }

            with open(train_file, "r", encoding="utf-8") as f:
                pdata = json.load(f)

            topics = pdata.get("topics", [])
            if not topics:
                return {
                    "participant": participant,
                    "warning": "No topics in train data",
                }

            # Prepare output directories
            out_dir = self.output_root / participant
            comments_dir = out_dir / "context_graphs"
            comments_dir.mkdir(parents=True, exist_ok=True)

            # Log LLM mode once per participant for debugging
            if self.llm is not None:
                logger.info(f"LLM mode: ON for participant {participant}")
            else:
                logger.info(f"LLM mode: OFF (heuristics) for participant {participant}")

            aggregated = nx.DiGraph()
            built = 0

            for topic in tqdm(topics, desc=f"{participant}"):
                comment_text = topic.get("comment_text", "").strip()
                stance_label = topic.get("stance_label", "").strip() or "UNKNOWN"
                comment_id = (
                    topic.get("comment_id")
                    or topic.get("post_id")
                    or f"comment_{built}"
                )

                if not comment_text:
                    continue

                # Prepare per-comment debug dir (also used for LLM debug)
                cdir = comments_dir / str(comment_id)
                cdir.mkdir(parents=True, exist_ok=True)
                G = self._build_comment_graph(
                    comment_text,
                    stance_label,
                    post_title=topic.get("post_title"),
                    scenario=topic.get("scenario_description"),
                    debug_dir=cdir,
                )
                # Print a sample of prompt and raw answer for the first LLM-enabled comment
                if built == 0 and self.llm is not None:
                    try:
                        # Re-run minimal prompt just to capture text shown (since _build_comment_graph doesn't return it)
                        ctx_preview = self.llm.generate_context(
                            post_title=topic.get("post_title") or "",
                            scenario=topic.get("scenario_description") or "",
                            comment=comment_text,
                            stance=stance_label,
                        )
                        prompt_str = ctx_preview.get("_prompt", "<prompt hidden>")
                        raw_str = ctx_preview.get("_raw", "<response hidden>")
                        from utils.logging import print_colored

                        print_colored("=== LLM Prompt (sample) ===", "green", bold=True)
                        print_colored(prompt_str, "green")
                        print_colored(
                            "=== LLM Response (sample) ===", "blue", bold=True
                        )
                        print_colored(raw_str[:1000], "blue")
                    except Exception:
                        pass

                # Save per-comment graph
                self._save_graph_as_json(G, cdir / "context_graph.json")
                mermaid = self._generate_mermaid(
                    G, f"Context Graph: {participant} | {comment_id}"
                )
                with open(cdir / "context_graph.mmd", "w", encoding="utf-8") as mf:
                    mf.write(mermaid)
                # Markdown preview for Mermaid
                with open(cdir / "context_graph.md", "w", encoding="utf-8") as mf:
                    mf.write("```mermaid\n")
                    mf.write(mermaid)
                    mf.write("\n```")

                # Merge into aggregated
                self._merge_into_aggregated(aggregated, G)
                built += 1

            # Save aggregated outputs
            if built > 0:
                self._save_graph_as_json(
                    aggregated, out_dir / "participant_context_graph.json"
                )
                mermaid_agg = self._generate_mermaid(
                    aggregated, f"Participant Context Graph: {participant}"
                )
                with open(
                    out_dir / "participant_context_graph.mmd", "w", encoding="utf-8"
                ) as mf:
                    mf.write(mermaid_agg)
                # Markdown preview for aggregated Mermaid
                with open(
                    out_dir / "participant_context_graph.md", "w", encoding="utf-8"
                ) as mf:
                    mf.write("```mermaid\n")
                    mf.write(mermaid_agg)
                    mf.write("\n```")

            metadata = {
                "participant": participant,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "train_topics": len(topics),
                "graphs_built": built,
                "llm_mode": self.llm is not None,
            }
            with open(
                out_dir / "context_graph_metadata.json", "w", encoding="utf-8"
            ) as mf:
                json.dump(metadata, mf, indent=2)

            return {"participant": participant, "graphs_built": built}

        except Exception as e:
            return {"participant": participant, "error": str(e)}

    def process_all(self) -> Dict[str, Any]:
        """Process all participants under the dataset directory."""
        if self.dataset_dir.exists():
            participants = [p.name for p in self.dataset_dir.iterdir() if p.is_dir()]
        else:
            participants = []

        if not participants and DataPreprocessor is not None:
            try:
                participants = DataPreprocessor(
                    self.dataset_dir.as_posix()
                ).participants
            except Exception:
                participants = []

        stats = {"total": len(participants), "successful": 0, "failed": 0}
        for participant in participants:
            print(f"Building context graphs for {participant}...")
            result = self.process_participant(participant)
            if "error" in result:
                print(f"  ❌ {participant}: {result['error']}")
                stats["failed"] += 1
            else:
                print(f"  ✅ {participant}: {result.get('graphs_built', 0)} graphs")
                stats["successful"] += 1
        return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build context graphs from train comments"
    )
    parser.add_argument(
        "--dataset",
        default="data/30ppl_eval_v1_0728",
        help="Path to dataset directory (method-relative)",
    )
    parser.add_argument(
        "--out",
        default="intermediate_data/30ppl_eval_v1_0728",
        help="Output root for intermediate data (method-relative)",
    )
    parser.add_argument(
        "--participant", default=None, help="Specific participant to process"
    )
    parser.add_argument(
        "--llm_cfg",
        default=None,
        help="YAML config filename to enable Qwen LLM extraction (e.g., qwen_context_default.yaml)",
    )

    args = parser.parse_args()

    llm = None
    if args.llm_cfg and ContextLLMConfig is not None and QwenMaxContextLLM is not None:
        try:
            cfg = ContextLLMConfig(config_file=args.llm_cfg)
            llm = QwenMaxContextLLM(cfg)
            logger.info(
                f"Using Qwen LLM: model={cfg.model_name}, temp={cfg.temperature}, prompt={cfg.prompt_version}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM from config {args.llm_cfg}: {e}")
            llm = None

    builder = ContextGraphBuilder(
        dataset_dir=args.dataset, output_root=args.out, llm=llm
    )
    if args.participant:
        res = builder.process_participant(args.participant)
        print(json.dumps(res, indent=2))
    else:
        stats = builder.process_all()
        print(json.dumps(stats, indent=2))
