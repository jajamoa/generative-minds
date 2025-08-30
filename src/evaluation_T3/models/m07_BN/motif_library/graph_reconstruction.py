"""
Motif-Based Causal Graph Reconstruction

This module implements an algorithm to reconstruct causal graphs using motifs as building blocks.
The reconstruction starts from a seed node (typically upzoning_stance) and grows the graph
by iteratively adding best-matching motifs.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Set, List, Tuple, Optional, Union
from .motif_library import MotifLibrary, get_demographic_statistics
from .semantic_similarity import SemanticSimilarityEngine
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import argparse
import json


# Utils for label handling and constants
from .graph_reconstruction_utils import (
    sanitize_label,
    canonicalize_seed_label,
    is_stance_like_label,
)


class MotifBasedReconstructor:
    def __init__(
        self,
        motif_library: MotifLibrary,
        similarity_threshold: float = 0.4,
        node_merge_threshold: float = 0.8,
        target_demographic: Union[str, Dict, List] = None,
        demographic_weight: float = 0.3,
    ):
        """
        Initialize the reconstructor.

        Args:
            motif_library: Loaded MotifLibrary instance
            similarity_threshold: Threshold for semantic similarity matching
            node_merge_threshold: Threshold for merging similar nodes
            target_demographic: Target demographic(s) - can be string, dict, or list
            demographic_weight: Weight for demographic scoring
        """
        self.motif_library = motif_library
        self.similarity_threshold = similarity_threshold
        self.node_merge_threshold = node_merge_threshold
        self.similarity_engine = SemanticSimilarityEngine(use_wordnet=True)
        self.target_demographic = target_demographic
        self.demographic_weight = demographic_weight
        self.demographic_scores = self._calculate_demographic_scores()

    def _calculate_demographic_scores(self):
        """Calculate demographic similarity scores with finer-grained feature matching."""
        scores = {}

        # if no demographic information, return empty dict
        if not self.motif_library.demographic_distribution:
            return scores

        # if no target demographic or format is wrong, return empty dict
        if not self.target_demographic or not isinstance(self.target_demographic, dict):
            return scores

        # get all features of target demographic
        target_features = set()
        for key, value in self.target_demographic.items():
            if isinstance(value, (list, set)):
                target_features.update(value)
            else:
                target_features.add(str(value))

        for demo, count in self.motif_library.demographic_distribution.items():
            frequency_score = 1.0 - (
                count / sum(self.motif_library.demographic_distribution.values())
            )
            demo_features = set(str(demo).split("_"))
            matching_features = len(target_features.intersection(demo_features))
            feature_score = matching_features / max(
                len(target_features), len(demo_features)
            )
            scores[demo] = 0.3 * frequency_score + 0.7 * feature_score

        return scores

    def _demographic_similarity(
        self, demo_of_motif: str, target_demo: Union[str, Dict, List]
    ) -> float:
        """Calculate similarity between motif's demographic and target demographic(s).

        Args:
            demo_of_motif: Demographic of the motif
            target_demo: Target demographic(s) - can be string, dict, or list

        Returns:
            float: Similarity score between 0 and 1
        """
        NONSENSE_DEMOGRAPHICS = ["unknown", "other", "", " ", "none", None]

        if demo_of_motif in NONSENSE_DEMOGRAPHICS:
            return 0.0

        # if target_demo is a dict (full agent profile)
        if isinstance(target_demo, dict):
            # extract key demographic features
            key_features = target_demo

            # calculate similarity for each feature
            similarities = []
            for feature, value in key_features.items():
                if (
                    feature in demo_of_motif
                ):  # if motif's demographic contains this feature
                    if feature == "age":
                        # age difference calculation
                        try:
                            age_diff = abs(
                                int(value) - int(demo_of_motif.split("_")[1])
                            )
                            similarities.append(
                                max(0, 1 - age_diff / 50)
                            )  # 50 years difference considered completely different
                        except:
                            similarities.append(
                                0.5
                            )  # if cannot compare, give medium score
                    else:
                        # exact match for other features
                        similarities.append(1.0 if str(value) in demo_of_motif else 0.0)

            return sum(similarities) / len(similarities) if similarities else 0.0

        # if target_demo is a list (multiple demographic features)
        elif isinstance(target_demo, list):
            # Calculate similarity with each target demographic, take max
            similarities = [
                self._demographic_similarity(demo_of_motif, single_demo)
                for single_demo in target_demo
                if single_demo not in NONSENSE_DEMOGRAPHICS
            ]
            return max(similarities) if similarities else 0.0

        # if target_demo is a string (single demographic feature)
        else:
            if target_demo in NONSENSE_DEMOGRAPHICS:
                return 0.0

            if demo_of_motif == target_demo:
                return 1.0

            demo_features = demo_of_motif.split("_")
            target_features = str(target_demo).split("_")

            matching_features = sum(
                1
                for f1 in demo_features
                for f2 in target_features
                if f1.lower() == f2.lower()
            )

            return matching_features / max(len(demo_features), len(target_features))

    def _similarity_between_demographics(
        self, motif_demo: Optional[Dict], target_demo: Optional[Union[Dict, List, str]]
    ) -> float:
        if not motif_demo or not target_demo:
            return 0.0

        if isinstance(target_demo, list):
            sims = [
                self._similarity_between_demographics(motif_demo, td)
                for td in target_demo
            ]
            return max(sims) if sims else 0.0

        if isinstance(target_demo, str):
            try:
                for v in motif_demo.values():
                    if isinstance(v, (list, set, tuple)):
                        if any(str(target_demo).lower() in str(x).lower() for x in v):
                            return 1.0
                    else:
                        if str(target_demo).lower() in str(v).lower():
                            return 1.0
            except Exception:
                return 0.0
            return 0.0

        if isinstance(target_demo, dict):
            similarities: List[float] = []
            for key, tval in target_demo.items():
                mval = motif_demo.get(key)
                if mval is None:
                    continue
                if key.lower() == "age":
                    try:
                        age_t = float(tval)
                        age_m = float(mval)
                        diff = abs(age_t - age_m)
                        similarities.append(max(0.0, 1.0 - diff / 50.0))
                    except Exception:
                        similarities.append(1.0 if str(tval) == str(mval) else 0.0)
                else:
                    if isinstance(tval, (list, set, tuple)):
                        tset = {str(x).lower() for x in tval}
                        if isinstance(mval, (list, set, tuple)):
                            mset = {str(x).lower() for x in mval}
                        else:
                            mset = {str(mval).lower()}
                        inter = tset & mset
                        denom = len(tset | mset) or 1
                        similarities.append(len(inter) / denom)
                    else:
                        similarities.append(
                            1.0 if str(tval).lower() == str(mval).lower() else 0.0
                        )

            return sum(similarities) / len(similarities) if similarities else 0.0

        return 0.0

    def _compute_motif_demographic_similarity(
        self, motif: nx.DiGraph, group_key: Optional[str]
    ) -> float:
        motif_demo = None
        try:
            motif_demo = getattr(motif, "graph", {}).get("demographics", {})
        except Exception:
            motif_demo = {}

        if motif_demo:
            sim = self._similarity_between_demographics(
                motif_demo, self.target_demographic
            )
            return sim

        if group_key and group_key in self.motif_library.motif_metadata:
            demos_list = self.motif_library.motif_metadata[group_key].get(
                "demographics_list", []
            )
            if isinstance(demos_list, list) and demos_list:
                sims = [
                    self._similarity_between_demographics(d, self.target_demographic)
                    for d in demos_list
                    if isinstance(d, dict)
                ]
                return max(sims) if sims else 0.0

        return 0.0

    def _debug_motif(
        self,
        motif: nx.DiGraph,
        group_key: Optional[str],
        reason: str,
        target_label: str,
        semantic_sim: float,
        structural_score: Optional[float] = None,
        coverage_score: Optional[float] = None,
        demo_sim: Optional[float] = None,
    ):
        try:
            nodes = list(motif.nodes())
            labels = {n: motif.nodes[n].get("label", str(n)) for n in nodes}
            print(
                f"[DEBUG] {reason}: group={group_key}, nodes={nodes}, labels={labels}, target='{target_label}', semSim={semantic_sim:.3f}, struct={structural_score}, cover={coverage_score}, demoSim={demo_sim}"
            )
        except Exception:
            pass

    def find_motif_candidates(
        self, frontier_node: str, covered_nodes: Set[str], current_graph: nx.DiGraph
    ) -> List[Tuple[nx.DiGraph, float, str]]:
        """
        Find candidate motifs that could extend from a frontier node.
        Now includes semantic matching for finding more relevant motifs.
        """
        candidates = []

        # Get frontier node label
        frontier_label = current_graph.nodes[frontier_node].get("label", frontier_node)

        # Prefer higher-quality groups (more instances, broader demographics)
        sorted_groups = sorted(
            self.motif_library.motif_metadata.items(),
            key=lambda kv: (
                -kv[1].get("instances", 0),
                -len(kv[1].get("demographics_list", [])),
            ),
        )

        for group_key, meta in sorted_groups:
            motifs = self.motif_library.semantic_motifs.get(group_key, [])
            for motif in motifs:
                # Try matching each node in the motif to the frontier node
                for motif_node in motif.nodes():
                    motif_label = motif.nodes[motif_node].get("label", "")

                    # Calculate semantic similarity
                    similarity = self.similarity_engine.node_similarity(
                        motif_label, frontier_label
                    )

                    if similarity >= self.similarity_threshold:
                        # Calculate extendability score
                        extendability = self._calculate_extendability(
                            motif, motif_node, current_graph, frontier_node
                        )

                        # Include demographic similarity (per-motif only here)
                        demographic_sim = self._compute_motif_demographic_similarity(
                            motif, None
                        )
                        base_score = 0.6 * similarity + 0.4 * extendability
                        demo_w = max(
                            0.0, min(1.0, getattr(self, "demographic_weight", 0.3))
                        )
                        score = (1.0 - demo_w) * base_score + demo_w * demographic_sim
                        if score < self.similarity_threshold:  # very weak overall
                            # self._debug_motif(
                            #     motif,
                            #     None,
                            #     "Skip-low-score",
                            #     frontier_label,
                            #     similarity,
                            #     extendability,
                            #     None,
                            #     demographic_sim,
                            # )
                            continue
                        candidates.append((motif, score, motif_node))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _calculate_extendability(
        self,
        motif: nx.DiGraph,
        motif_node: str,
        current_graph: nx.DiGraph,
        frontier_node: str,
    ) -> float:
        """
        Calculate how extendable a motif is from the frontier node.

        Returns:
            Float score between 0-1 indicating extendability
        """
        # Count potential new connections
        in_edges = motif.in_degree(motif_node)
        out_edges = motif.out_degree(motif_node)

        current_in = current_graph.in_degree(frontier_node)
        current_out = current_graph.out_degree(frontier_node)

        # Score based on how many new edges could be added
        potential_new_edges = max(0, in_edges - current_in) + max(
            0, out_edges - current_out
        )

        # Normalize score
        max_possible = motif.number_of_edges()
        return potential_new_edges / max_possible if max_possible > 0 else 0

    def _should_merge_nodes(self, label1: str, label2: str) -> bool:
        """
        Determine if two nodes should be merged based on semantic similarity.
        """
        similarity = self.similarity_engine.node_similarity(label1, label2)
        return similarity >= self.node_merge_threshold

    def _find_mergeable_node(self, G: nx.DiGraph, new_label: str) -> Optional[str]:
        """
        Find existing node that can be merged with the new node.
        """
        for node in G.nodes():
            existing_label = G.nodes[node].get("label", "")
            if self._should_merge_nodes(new_label, existing_label):
                return node
        return None

    def _integrate_motif(
        self,
        G: nx.DiGraph,
        motif: nx.DiGraph,
        anchor_node: str,
        covered_nodes: Set[str],
    ) -> None:
        """
        Integrate a motif into the growing graph with node merging.
        """
        # Create mapping for node integration
        node_mapping = {}

        # First, handle the anchor node
        node_mapping[anchor_node] = anchor_node

        # Process each node in the motif
        for node in motif.nodes():
            if node in node_mapping:
                continue

            new_label = motif.nodes[node].get("label", "")

            # Try to find a mergeable node
            mergeable_node = self._find_mergeable_node(G, new_label)

            if mergeable_node:
                # Merge with existing node
                node_mapping[node] = mergeable_node
            else:
                # Add as new node
                G.add_node(node, label=new_label)
                node_mapping[node] = node

        # Add edges using the node mapping
        for u, v in motif.edges():
            mapped_u = node_mapping[u]
            mapped_v = node_mapping[v]
            if not G.has_edge(mapped_u, mapped_v):
                G.add_edge(mapped_u, mapped_v)

        # Update covered nodes
        covered_nodes.update(node_mapping.values())

    # =============================
    # GROW PHASE: find and integrate motifs
    # =============================
    def _grow_graph(
        self,
        target_node: str,
        max_iterations: int,
        min_score: float,
        k: int,
    ) -> nx.DiGraph:
        """
        GROW: Build an initial graph by iteratively adding best-matching motifs starting
        from the seed node.
        """
        G = nx.DiGraph()
        seed_label = canonicalize_seed_label(target_node)
        G.add_node(target_node, label=seed_label)
        covered_nodes = {target_node}

        # Phase 1: Initial growth from target_node
        print("Phase 1: Growing from target node…")
        candidates = self.find_motif_candidates_reverse(target_node, covered_nodes, G)
        if not candidates:
            print("No reverse candidates found; trying forward candidates from stance…")
            forward = self.find_motif_candidates(target_node, covered_nodes, G)
            candidates = [(m, s, n, None, "unknown") for (m, s, n) in forward]
        print(
            f"Reverse candidates: {len(candidates)} (top5 scores: "
            f"{[round(c[1],3) for c in sorted(candidates, key=lambda x: x[1], reverse=True)[:5]]})"
        )

        candidates.sort(key=lambda x: x[1], reverse=True)
        initial_motifs = candidates[:k]
        print(f"Found {len(initial_motifs)} initial motifs to add")
        for motif, score, motif_node, _, _ in initial_motifs:
            self._integrate_motif_reverse(
                G,
                motif,
                target_node,
                motif_node,
                covered_nodes,
                block_outgoing_from_target=True,
            )
        print(f"After Phase 1: {len(G.nodes())} nodes, {len(G.edges())} edges")

        # Phase 2: Expand from other nodes
        print("Phase 2: Growing from other nodes…")
        frontier = set(G.nodes()) - {target_node}
        max_nodes = 15
        iteration = 0
        while frontier and iteration < max_iterations and len(G.nodes()) < max_nodes:
            iteration += 1
            print(f"Iteration {iteration}, Current frontier size: {len(frontier)}")
            new_frontier = set()
            for f_node in list(frontier):
                candidates = self.find_motif_candidates_reverse(
                    f_node, covered_nodes, G
                )
                for motif, score, motif_node, group_key, _ in sorted(
                    candidates, key=lambda x: x[1], reverse=True
                )[:k]:
                    if score < min_score:
                        continue

                    new_nodes_count = len(set(motif.nodes()) - covered_nodes)
                    if len(G.nodes()) + new_nodes_count > max_nodes:
                        continue

                    old_nodes = set(G.nodes())
                    self._integrate_motif_reverse(
                        G,
                        motif,
                        f_node,
                        motif_node,
                        covered_nodes,
                        block_outgoing_from_target=False,
                    )

                    if set(G.nodes()) == old_nodes:
                        continue

                    added_nodes = set(G.nodes()) - old_nodes
                    new_frontier.update(
                        node
                        for node in added_nodes
                        if node != target_node and G.in_degree(node) == 0
                    )
                    break

            frontier = new_frontier
            print(
                f"After iteration {iteration}: {len(G.nodes())} nodes, {len(G.edges())} edges"
            )

        return G

    # =============================
    # PRUNING PHASE: reduce duplicates and sanitize
    # =============================
    def _merge_duplicate_label_nodes(self, G: nx.DiGraph, target_node: str) -> None:
        """
        Merge nodes that share the exact same sanitized label. Prefer keeping
        the target_node as representative for stance-like labels.
        """
        label_to_nodes: Dict[str, List[str]] = {}
        for node in list(G.nodes()):
            label = G.nodes[node].get("label", "")
            label_to_nodes.setdefault(label, []).append(node)

        for label, nodes in label_to_nodes.items():
            if len(nodes) <= 1:
                continue

            # Choose representative
            rep = None
            if label == self.target_node and target_node in nodes:
                rep = target_node
            else:
                rep = nodes[0]

            for dup in nodes:
                if dup == rep:
                    continue

                # Move incoming edges to representative
                for src, _ in list(G.in_edges(dup)):
                    if src != rep and not G.has_edge(src, rep):
                        data = G.get_edge_data(src, dup) or {}
                        modifier = data.get("modifier", 1.0)
                        G.add_edge(src, rep, modifier=modifier)

                # Move outgoing edges to representative
                for _, dst in list(G.out_edges(dup)):
                    if dst != rep and not G.has_edge(rep, dst):
                        data = G.get_edge_data(dup, dst) or {}
                        modifier = data.get("modifier", 1.0)
                        G.add_edge(rep, dst, modifier=modifier)

                # Finally remove duplicate node
                if dup in G:
                    G.remove_node(dup)

    def _prune_graph(self, G: nx.DiGraph, target_node: str) -> nx.DiGraph:
        """
        PRUNING: Sanitize labels, coalesce stance-like duplicates, remove
        invalid edges, and enforce connectivity to the stance node.
        """
        print("Cleaning up (PRUNING)…")

        # 1) Sanitize all labels
        for node in G.nodes():
            old_label = G.nodes[node].get("label", "")
            G.nodes[node]["label"] = sanitize_label(old_label)

        # Standardize target label
        G.nodes[target_node]["label"] = self.target_node

        # 2) Merge stance-like nodes into the stance target
        nodes_to_merge = []
        for node in G.nodes():
            if node == target_node:
                continue
            label = G.nodes[node].get("label", "")
            if is_stance_like_label(label):
                nodes_to_merge.append(node)

        if nodes_to_merge:
            print(f"Merging {len(nodes_to_merge)} additional upzoning stance nodes")
            for node in nodes_to_merge:
                incoming = list(G.in_edges(node))
                for src, _ in incoming:
                    if not G.has_edge(src, target_node):
                        G.add_edge(src, target_node, modifier=1.0)
                if node in G:
                    G.remove_node(node)

        # 3) Remove any outgoing edges from stance node
        outgoing_edges = list(G.out_edges(target_node))
        if outgoing_edges:
            print(
                f"Removing {len(outgoing_edges)} invalid outgoing edges from {target_node}"
            )
            for src, dst in outgoing_edges:
                if G.has_edge(src, dst):
                    G.remove_edge(src, dst)

        # 4) Merge nodes with identical labels (conservative)
        self._merge_duplicate_label_nodes(G, target_node)

        # 5) Final connectivity check
        for node in list(G.nodes()):
            if node != target_node and not nx.has_path(G, node, target_node):
                G.add_edge(node, target_node, modifier=1.0)

        return G

    def reconstruct_graph(
        self,
        target_node: str,
        max_iterations: int = 100,
        min_score: float = 0.3,
        k: int = 3,
    ) -> nx.DiGraph:
        """
        Entry point: orchestrates GROW (motif discovery/integration) and PRUNING
        (deduplication and sanitation) to produce the final graph.
        """
        self.target_node = target_node
        print("Starting graph reconstruction…")
        G = self._grow_graph(target_node, max_iterations, min_score, k)
        G = self._prune_graph(G, target_node)
        print(f"Final graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
        print(f"Stance node ({target_node}) out degree: {G.out_degree(target_node)}")
        return G

    def find_motif_candidates_reverse(
        self, target_node: str, covered_nodes: Set[str], current_graph: nx.DiGraph
    ) -> List[Tuple[nx.DiGraph, float, str, str, str]]:
        """
        Find candidate motifs where target node is the outcome.
        """
        candidates = []
        target_label = current_graph.nodes[target_node].get("label", target_node)

        # Debug print
        print(f"Finding candidates for target node: {target_label}")

        sorted_groups = sorted(
            self.motif_library.motif_metadata.items(),
            key=lambda kv: (
                -kv[1].get("instances", 0),
                -len(kv[1].get("demographics_list", [])),
            ),
        )

        for group_key, meta in sorted_groups:
            motifs = self.motif_library.semantic_motifs.get(group_key, [])
            for motif in motifs:
                # For each node in the motif
                for motif_node in motif.nodes():
                    motif_label = motif.nodes[motif_node].get("label", "")

                    # Calculate semantic similarity with an upzoning keyword boost
                    semantic_sim = self.similarity_engine.node_similarity(
                        motif_label, target_label
                    )
                    if (
                        "upzoning" in motif_label.lower()
                        and "upzoning" in target_label.lower()
                    ):
                        semantic_sim = max(semantic_sim, 0.5)

                    if semantic_sim >= self.similarity_threshold:
                        # If this is upzoning_stance, ensure it has no out-degree
                        if "upzoning_stance" in motif_label.lower():
                            if motif.out_degree(motif_node) > 0:
                                continue

                        # Calculate scores
                        structural_score = self._calculate_structural_fit(
                            motif, motif_node, current_graph, target_node
                        )
                        coverage_score = self._calculate_coverage_score(
                            motif, covered_nodes
                        )

                        # Demographic similarity
                        demographic_sim = self._compute_motif_demographic_similarity(
                            motif, group_key
                        )
                        # # Debug demographics insight
                        # try:
                        #     demos_list = motif.graph.get("demographics_list", [])
                        #     srcs = motif.graph.get("sources", [])
                        #     if demos_list:
                        #         print(
                        #             f"[DEBUG] motif group={group_key} has {len(demos_list)} demographics sources={len(srcs)} demoSim={demographic_sim:.3f}"
                        #         )
                        # except Exception:
                        #     pass

                        # Calculate final score (blend base and demographics)
                        base_score = (
                            0.4 * semantic_sim
                            + 0.3 * structural_score
                            + 0.3 * coverage_score
                        )
                        demo_w = max(
                            0.0, min(1.0, getattr(self, "demographic_weight", 0.3))
                        )
                        final_score = (
                            1.0 - demo_w
                        ) * base_score + demo_w * demographic_sim

                        # Require demographic support if target contains a demographic profile
                        # If there is a non-empty dict target_demographic, enforce a minimum demographic_sim
                        if (
                            isinstance(self.target_demographic, dict)
                            and self.target_demographic
                        ):
                            if demographic_sim < 0.2:
                                continue

                        # Get demographic info
                        motif_demographic = (
                            motif.graph.get("demographic", "unknown")
                            if hasattr(motif, "graph")
                            else "unknown"
                        )

                        candidates.append(
                            (
                                motif,
                                final_score,
                                motif_node,
                                group_key,
                                motif_demographic,
                            )
                        )

        # Debug print
        print(f"Found {len(candidates)} candidates")
        return candidates

    def _calculate_structural_fit(
        self,
        motif: nx.DiGraph,
        motif_node: str,
        current_graph: nx.DiGraph,
        target_node: str,
    ) -> float:
        """
        Calculate structural fit score for reverse building.
        """
        # Count incoming edges that could be added
        motif_in = motif.in_degree(motif_node)
        current_in = current_graph.in_degree(target_node)

        # Score based on potential new incoming connections
        potential_new_edges = max(0, motif_in - current_in)

        # Normalize score
        max_possible = motif.number_of_edges()
        return potential_new_edges / max_possible if max_possible > 0 else 0

    def _calculate_coverage_score(
        self,
        motif: nx.DiGraph,
        covered_nodes: Set[str],
    ) -> float:
        """
        Calculate coverage score for reverse building.
        """
        # Count nodes in motif that are not covered
        uncovered_nodes = set(motif.nodes()) - covered_nodes
        uncovered_count = len(uncovered_nodes)

        # Normalize score
        max_possible = motif.number_of_nodes()
        return uncovered_count / max_possible if max_possible > 0 else 0

    def _integrate_motif_reverse(
        self,
        G: nx.DiGraph,
        motif: nx.DiGraph,
        target_node: str,
        motif_node: str,
        covered_nodes: Set[str],
        block_outgoing_from_target: bool = True,
    ) -> None:
        """
        Integrate a motif into the graph while ensuring path connectivity to target node.
        The target_node (upzoning_stance) should never have outgoing edges.

        Args:
            G: The growing graph
            motif: The motif to integrate
            target_node: The target node (usually upzoning_stance)
            motif_node: The node in the motif that matches the target_node
            covered_nodes: Set of already covered nodes
        """
        # Create a copy of the motif to modify
        motif_copy = motif.copy()

        # Optionally remove edges where motif_node is source (only for stance seed)
        if block_outgoing_from_target:
            outgoing_edges = list(motif_copy.out_edges(motif_node))
            for u, v in outgoing_edges:
                motif_copy.remove_edge(u, v)

        node_mapping = {}
        node_mapping[target_node] = target_node

        # Step 1: First map the target node and its direct predecessors
        target_predecessors = list(motif_copy.predecessors(motif_node))
        for pred in target_predecessors:
            new_label = (
                motif_copy.nodes[pred].get("label", "").replace(" ", "_").lower()
            )

            # Try to find mergeable node that has a path to target
            mergeable = None
            max_similarity = 0

            for existing_node in G.nodes():
                if existing_node == target_node:  # Skip comparing with target node
                    continue
                existing_label = G.nodes[existing_node].get("label", "").lower()
                similarity = self.similarity_engine.node_similarity(
                    new_label, existing_label
                )

                if (
                    similarity > max_similarity
                    and similarity >= self.node_merge_threshold
                ):
                    max_similarity = similarity
                    mergeable = existing_node

            if mergeable:
                node_mapping[pred] = mergeable
            else:
                new_node = f"n{len(G.nodes()) + 1}"
                G.add_node(new_node, label=new_label)
                node_mapping[pred] = new_node
                G.add_edge(new_node, target_node, modifier=1.0)

        # Step 2: Map remaining nodes
        remaining_nodes = [n for n in motif_copy.nodes() if n not in node_mapping]
        for node in remaining_nodes:
            new_label = motif_copy.nodes[node].get("label", "")

            # Try to find mergeable node
            mergeable = None
            max_similarity = 0

            for existing_node in G.nodes():
                if existing_node == target_node:  # Skip comparing with target node
                    continue
                existing_label = G.nodes[existing_node].get("label", "")
                similarity = self.similarity_engine.node_similarity(
                    new_label, existing_label
                )

                if (
                    similarity > max_similarity
                    and similarity >= self.node_merge_threshold
                ):
                    max_similarity = similarity
                    mergeable = existing_node

            if mergeable:
                node_mapping[node] = mergeable
            else:
                new_node = f"n{len(G.nodes()) + 1}"
                G.add_node(new_node, label=new_label)
                node_mapping[node] = new_node

        # Step 3: Add edges
        for u, v in motif_copy.edges():
            mapped_u = node_mapping[u]
            mapped_v = node_mapping[v]

            # Never add edges where the current target is the source only if blocking is enabled
            if block_outgoing_from_target and mapped_u == target_node:
                continue

            if not G.has_edge(mapped_u, mapped_v):
                # Check if adding this edge would create a cycle
                if not nx.has_path(G, mapped_v, mapped_u):
                    # Get edge attributes
                    edge_attrs = motif_copy.edges[u, v]
                    if "modifier" not in edge_attrs:
                        edge_attrs["modifier"] = 1.0

                    # Add the edge
                    G.add_edge(mapped_u, mapped_v, **edge_attrs)

        # Step 4: Ensure all nodes have a path to target
        for node in G.nodes():
            if node != target_node and not nx.has_path(G, node, target_node):
                # Find the closest node that has a path to target
                best_intermediate = None
                best_similarity = -1

                for intermediate in G.nodes():
                    if (
                        intermediate != node
                        and intermediate != target_node
                        and nx.has_path(G, intermediate, target_node)
                    ):
                        similarity = self.similarity_engine.node_similarity(
                            G.nodes[node].get("label", ""),
                            G.nodes[intermediate].get("label", ""),
                        )
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_intermediate = intermediate

                if best_intermediate:
                    G.add_edge(node, best_intermediate, modifier=1.0)
                else:
                    # If no good intermediate found, connect directly to target
                    G.add_edge(node, target_node, modifier=1.0)

        # Update covered nodes
        covered_nodes.update(node_mapping.values())

    def update_frontier(self, G: nx.DiGraph, covered_nodes: Set[str]) -> Set[str]:
        """
        Update frontier to include only nodes that could have incoming edges.
        """
        frontier = set()

        # Add nodes that have incoming edges but might accept more
        for node in covered_nodes:
            # Only add to frontier if the node isn't upzoning_stance
            if node != "upzoning_stance":
                # Look for nodes that point to current node
                predecessors = set(G.predecessors(node))
                frontier.update(predecessors - covered_nodes)

        return frontier

    def save_as_json(
        self, G: nx.DiGraph, output_dir: str, filename: str = "reconstructed_graph.json"
    ) -> str:
        """
        Save the reconstructed graph as a JSON file.

        Args:
            G: Reconstructed NetworkX DiGraph
            output_dir: Directory to save the file
            filename: Output filename

        Returns:
            Path to saved JSON file
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        # Create JSON structure
        graph_data = {
            "nodes": {},
            "edges": {},
            "metadata": {
                "description": "Reconstructed causal graph from motifs",
                "node_count": len(G.nodes()),
                "edge_count": len(G.edges()),
            },
        }

        # Add nodes
        for i, node in enumerate(G.nodes()):
            node_id = str(node)
            graph_data["nodes"][node_id] = {
                "id": node_id,
                "label": G.nodes[node].get("label", str(node)),
                "type": "concept",
                "in_degree": G.in_degree(node),
                "out_degree": G.out_degree(node),
            }

        # Add edges with modifiers
        for i, (source, target) in enumerate(G.edges()):
            edge_id = f"e{i}"
            edge_data = G.edges[source, target]
            graph_data["edges"][edge_id] = {
                "id": edge_id,
                "source": str(source),
                "target": str(target),
                "type": "causal",
                "modifier": edge_data.get(
                    "modifier", 1.0
                ),  # Default to positive influence
            }

        # Save to file
        with open(output_path, "w") as f:
            json.dump(graph_data, f, indent=2)

        return output_path

    def save_as_mmd(
        self, G: nx.DiGraph, output_dir: str, filename: str = "reconstructed_graph.mmd"
    ) -> str:
        """
        Save the reconstructed graph as a Mermaid markdown file.
        Includes positive/negative causal relationships and link styling.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        # Create Mermaid markdown content
        md_lines = ["```mermaid"]
        mmd_lines = ["flowchart TD"]

        # Add nodes with labels
        for node in G.nodes():
            label = G.nodes[node].get("label", str(node))
            # Clean label for Mermaid compatibility
            clean_label = label.replace(" ", "_").replace("-", "_")
            mmd_lines.append(f"    {node}[{clean_label}]")

        # Add edges with different styles for positive/negative relationships
        edge_styles = []
        for i, (source, target) in enumerate(G.edges()):
            # For demonstration, we'll alternate between positive and negative relationships
            # In a real implementation, this should be based on edge attributes
            if i % 2 == 0:
                # Positive causal relationship
                mmd_lines.append(f"    {source} --> {target}")
                edge_styles.append("stroke:#00AA00,stroke-width:2px")
            else:
                # Negative causal relationship
                mmd_lines.append(f"    {source} --x {target}")
                edge_styles.append("stroke:#FF0000,stroke-dasharray:3,stroke-width:2px")

        # Add metadata as comments
        mmd_lines.insert(1, "%% Reconstructed Causal Graph")
        mmd_lines.insert(2, f"%% Nodes: {len(G.nodes())}")
        mmd_lines.insert(3, f"%% Edges: {len(G.edges())}")

        # Add link styles
        for i, style in enumerate(edge_styles):
            mmd_lines.append(f"    linkStyle {i} {style}")

        md_lines = md_lines + mmd_lines + ["```"]

        # Save both .mmd and .md files
        with open(output_path, "w") as f:
            f.write("\n".join(mmd_lines))

        # Save markdown version
        md_path = output_path.replace(".mmd", ".md")
        with open(md_path, "w") as f:
            f.write("\n".join(md_lines))

        return output_path

    def save_as_png(
        self, G: nx.DiGraph, output_dir: str, filename: str = "reconstructed_graph.png"
    ) -> str:
        """
        Save the reconstructed graph as a PNG image using NetworkX + Matplotlib.

        Nodes are labeled; stance node is highlighted. Positive edges are solid; negative dashed.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)

        # Node colors: stance node highlighted
        node_colors = []
        for node in G.nodes():
            label = G.nodes[node].get("label", "")
            if isinstance(label, str) and "upzoning_stance" in label.lower():
                node_colors.append("#FFD166")  # amber
            else:
                node_colors.append("#8ECAE6")  # light blue

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=900, alpha=0.9)

        # Edges: style by modifier
        solid_edges = []
        dashed_edges = []
        for u, v, data in G.edges(data=True):
            if data.get("modifier", 1.0) >= 0:
                solid_edges.append((u, v))
            else:
                dashed_edges.append((u, v))

        if solid_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=solid_edges, width=2.0, edge_color="#2A9D8F"
            )
        if dashed_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=dashed_edges,
                width=2.0,
                edge_color="#E76F51",
                style="dashed",
            )

        labels = {n: G.nodes[n].get("label", str(n)) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return output_path


def main(args):
    """
    Example usage of the improved motif-based reconstruction.
    """
    # Load motif library
    # library = MotifLibrary.load_library(args.motif_library)

    # Create reconstructor with adjusted parameters
    # reconstructor = MotifBasedReconstructor(
    #     library,
    #     similarity_threshold=0.3,  # Lower threshold to find more matches
    #     node_merge_threshold=0.8,  # High threshold for merging to avoid false positives
    # )

    """Use demographic-aware motif reconstruction."""
    # First analyze demographic distribution
    samples_dir = os.path.dirname(args.motif_library).replace("output", "")
    demo_stats = get_demographic_statistics(samples_dir)

    print("Demographic Distribution:")
    print("-" * 40)
    for demo, info in demo_stats["distribution"].items():
        print(f"{demo}: {info['count']} samples ({info['percentage']:.1f}%)")
    print()

    # Load motif library
    library = MotifLibrary.load_library(args.motif_library)

    # Select target demographic (can be obtained from command line arguments)
    if hasattr(args, "target_demographic"):
        target_demographic = args.target_demographic
    else:
        # Default to using the most common demographic
        most_common = max(
            demo_stats["distribution"].items(), key=lambda x: x[1]["count"]
        )[0]
        target_demographic = most_common

    print(f"Target demographic: {target_demographic}")

    # Create demographic-aware reconstructor
    reconstructor = MotifBasedReconstructor(
        library,
        similarity_threshold=0.3,
        node_merge_threshold=0.8,
        target_demographic=target_demographic,
        demographic_weight=0.3,  # 30% weight for demographic scoring
    )

    # Reconstruct graph from seed
    reconstructed_graph = reconstructor.reconstruct_graph(
        args.seed_node,
        max_iterations=args.max_iterations,
        min_score=0.3,  # Lower score threshold to accept more candidates
    )

    # Analyze results
    print(f"Reconstructed graph has {len(reconstructed_graph.nodes())} nodes")
    print(f"Reconstructed graph has {len(reconstructed_graph.edges())} edges")

    # Create visualizations
    print("\nGenerating visualizations...")

    # Save graph as JSON
    json_path = reconstructor.save_as_json(reconstructed_graph, args.output_dir)
    print(f"Reconstructed graph saved as JSON to: {json_path}")

    # Save graph as MMD
    mmd_path = reconstructor.save_as_mmd(reconstructed_graph, args.output_dir)
    print(f"Reconstructed graph saved as MMD to: {mmd_path}")


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    args = argparse.ArgumentParser()
    args.add_argument("--seed_node", type=str, default="upzoning_stance")
    args.add_argument("--max_iterations", type=int, default=20)
    args.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(CURRENT_DIR, "graph_reconstruction"),
    )
    args.add_argument(
        "--motif_library",
        type=str,
        default=os.path.join(CURRENT_DIR, "motif_library_output", "motif_library.json"),
    )
    args.add_argument(
        "--target_demographic", type=str, help="Target demographic for reconstruction"
    )
    args.add_argument(
        "--demographic_weight",
        type=float,
        default=0.3,
        help="Weight for demographic scoring (0-1)",
    )
    args = args.parse_args()
    main(args)
