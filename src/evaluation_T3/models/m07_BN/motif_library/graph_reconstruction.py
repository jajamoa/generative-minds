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
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import argparse
import json


from .motif_library import MotifLibrary, get_demographic_statistics
from .semantic_similarity import SemanticSimilarityEngine


class MotifBasedReconstructor:
    def __init__(
        self,
        motif_library: MotifLibrary,
        similarity_threshold: float = 0.4,
        node_merge_threshold: float = 0.8,
        target_demographic: Dict = None,
        demographic_weight: float = 0.3,
    ):
        """
        Initialize the reconstructor with updated demographic handling.

        Args:
            motif_library: MotifLibrary instance containing motif patterns
            similarity_threshold: Threshold for node similarity
            node_merge_threshold: Threshold for merging nodes
            target_demographic: Dictionary containing demographic information matching library format
            demographic_weight: Weight given to demographic similarity in scoring
        """
        self.motif_library = motif_library

        # Debug information
        print("\nDebug: Motif Library Content")
        print(f"Type of motif_library: {type(motif_library)}")
        print(f"Available attributes: {dir(motif_library)}")

        if hasattr(motif_library, "semantic_motifs"):
            print("\nSemantic Motifs Content:")
            print(f"Type of semantic_motifs: {type(motif_library.semantic_motifs)}")
            print(
                f"Number of motif groups: {len(motif_library.semantic_motifs) if motif_library.semantic_motifs else 0}"
            )
            if motif_library.semantic_motifs:
                print(
                    "Available group keys:",
                    list(motif_library.semantic_motifs.keys())[:5],
                )
        else:
            print("No semantic_motifs attribute found!")

        # Rest of initialization
        self.similarity_threshold = similarity_threshold
        self.node_merge_threshold = node_merge_threshold
        self.target_demographic = target_demographic
        self.demographic_weight = demographic_weight
        self.similarity_engine = SemanticSimilarityEngine()

    def _demographic_similarity(self, motif: nx.DiGraph, target_demo: Dict) -> float:
        """
        Calculate demographic similarity between motif and target demographics.

        Args:
            motif: Motif graph with demographic metadata
            target_demo: Target demographic dictionary

        Returns:
            float: Similarity score between 0 and 1
        """
        if not target_demo or "metadata" not in motif.graph:
            return 0.0

        motif_demos = motif.graph.get("metadata", {}).get("demographics", [])
        if not motif_demos:
            return 0.0

        similarities = []
        for demo in motif_demos:
            if not isinstance(demo, dict) or "demographic" not in demo:
                continue

            demo_data = demo["demographic"]
            score = 0
            total_weight = 0

            # Define weights for different demographic features
            weights = {
                "age": 0.15,
                "income": 0.2,
                "householder type": 0.2,
                "occupation": 0.15,
                "Geo Mobility": 0.1,
                "marital status": 0.1,
                "has children under 18": 0.1,
            }

            for key, weight in weights.items():
                if key in target_demo and key in demo_data:
                    if key == "age":
                        # Age similarity within 10 years
                        age_diff = abs(int(target_demo[key]) - int(demo_data[key]))
                        feature_score = max(0, 1 - age_diff / 20)
                    elif key == "income":
                        feature_score = self._compare_income_ranges(
                            target_demo[key], demo_data[key]
                        )
                    elif key == "has children under 18":
                        feature_score = (
                            1.0 if target_demo[key] == demo_data[key] else 0.0
                        )
                    else:
                        # Exact match for categorical variables
                        feature_score = (
                            1.0 if target_demo[key] == demo_data[key] else 0.0
                        )

                    score += weight * feature_score
                    total_weight += weight

            if total_weight > 0:
                similarities.append(score / total_weight)

        # Return maximum similarity across all demographics in the motif
        return max(similarities) if similarities else 0.0

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

    def _compare_income_ranges(self, target_income: str, motif_income: str) -> float:
        """
        Compare income ranges and return similarity score
        """
        income_levels = [
            "0-$24,999",
            "$25,000-$49,999",
            "$50,000-$74,999",
            "$75,000-$99,999",
            "$100,000-$124,999",
            "$125,000-$149,999",
            "$150,000-$199,999",
            ">$200,000",
        ]

        try:
            target_idx = next(
                i for i, level in enumerate(income_levels) if level in target_income
            )
            motif_idx = next(
                i for i, level in enumerate(income_levels) if level in motif_income
            )
            return 1.0 - abs(target_idx - motif_idx) / len(income_levels)
        except:
            return 0.0

    def find_motif_candidates(
        self, frontier_node: str, covered_nodes: Set[str], current_graph: nx.DiGraph
    ) -> List[Tuple[nx.DiGraph, float, str]]:
        """
        Find candidate motifs that could extend from the frontier node.
        Updated to handle new motif format and scoring.
        """
        candidates = []

        # Get priority motif types based on current graph state
        priority_types = self._get_priority_motif_types()

        # Iterate through semantic motifs in library
        for group_key, group_data in self.motif_library.semantic_motifs.items():
            motif_type = group_data[0]["metadata"][
                "motif_type"
            ]  # Get type from first motif
            type_weight = self._calculate_type_weight(motif_type, priority_types)

            for motif_data in group_data:
                motif = nx.DiGraph()

                # Reconstruct motif from data
                for node in motif_data["nodes"]:
                    motif.add_node(
                        node,
                        label=motif_data["node_labels"][node],
                        confidence=motif_data["metadata"]["confidence"].get(node, 0.0),
                        importance=motif_data["metadata"]["importance"].get(node, 0.0),
                    )

                for edge in motif_data["edges"]:
                    motif.add_edge(edge[0], edge[1])

                # Add metadata
                motif.graph["metadata"] = motif_data["metadata"]

                # Calculate various scores
                structural_score = self._calculate_motif_structural_fit(
                    motif, current_graph
                )
                coverage_score = self._calculate_coverage_score(motif, covered_nodes)
                demo_score = (
                    self._demographic_similarity(motif, self.target_demographic)
                    if self.target_demographic
                    else 0.0
                )

                # Combined score with weights
                total_score = (
                    0.4 * structural_score + 0.3 * coverage_score + 0.3 * demo_score
                ) * type_weight

                if total_score >= self.similarity_threshold:
                    for node in motif.nodes():
                        # Check if this node could connect to frontier_node
                        if self._can_connect(node, frontier_node, motif, current_graph):
                            candidates.append((motif, total_score, node))

        return sorted(candidates, key=lambda x: x[1], reverse=True)

    def _can_connect(
        self, motif_node: str, graph_node: str, motif: nx.DiGraph, graph: nx.DiGraph
    ) -> bool:
        """
        Check if a motif node can connect to a graph node.
        """
        motif_label = motif.nodes[motif_node].get("label", "")
        graph_label = graph.nodes[graph_node].get("label", "")

        # Check semantic similarity
        if (
            self.similarity_engine.node_similarity(motif_label, graph_label)
            >= self.node_merge_threshold
        ):
            return True

        return False

    def _calculate_motif_structural_fit(
        self, motif: nx.DiGraph, current_graph: nx.DiGraph
    ) -> float:
        """
        Calculate how well the entire motif structure fits with the current graph
        """
        # 计算节点重叠
        motif_nodes = set(motif.nodes())
        graph_nodes = set(current_graph.nodes())
        overlap_ratio = len(motif_nodes & graph_nodes) / len(motif_nodes)

        # 计算边模式相似度
        motif_edge_count = motif.number_of_edges()
        graph_edge_count = current_graph.number_of_edges()
        edge_ratio = abs(motif_edge_count - graph_edge_count) / max(
            motif_edge_count, graph_edge_count
        )

        # 综合分数
        return 0.7 * (1 - overlap_ratio) + 0.3 * (1 - edge_ratio)

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

    def _get_priority_motif_types(self) -> Dict[str, float]:
        """
        Based on target demographics, determine which motif types should be prioritized
        """
        priorities = {}

        if not self.target_demographic:
            return priorities

        # 根据人口统计特征设置优先级
        if self.target_demographic.get("householder type") == "Owner-occupied":
            priorities["Economic_Owner"] = 1.0
            priorities["Family"] = 0.8
            priorities["Transportation_Car"] = 0.7
        elif self.target_demographic.get("householder type") == "Renter":
            priorities["Economic_Renter"] = 1.0
            priorities["Career"] = 0.8
            priorities["Transportation_Transit"] = 0.7

        if self.target_demographic.get("has children under 18"):
            priorities["Family"] = max(priorities.get("Family", 0), 0.9)

        if (
            self.target_demographic.get("means of transportation")
            == "Public Transportation"
        ):
            priorities["Transportation_Transit"] = max(
                priorities.get("Transportation_Transit", 0), 0.9
            )
        elif (
            self.target_demographic.get("means of transportation")
            == "Car / Truck / Van"
        ):
            priorities["Transportation_Car"] = max(
                priorities.get("Transportation_Car", 0), 0.9
            )

        # 收入相关优先级
        income = self.target_demographic.get("income", "")
        if any(high in income for high in [">$200,000", "$150,000"]):
            priorities["Economic_Owner"] = max(priorities.get("Economic_Owner", 0), 0.9)
        elif any(low in income for low in ["$50,000", "$75,000"]):
            priorities["Economic_Renter"] = max(
                priorities.get("Economic_Renter", 0), 0.9
            )

        return priorities

    def _calculate_type_weight(
        self, motif_type: str, priority_types: Dict[str, float]
    ) -> float:
        """
        Calculate weight for a motif type based on demographic priorities
        """
        return priority_types.get(motif_type, 0.1)

    def find_motif_candidates_reverse(
        self, target_node: str, covered_nodes: Set[str], current_graph: nx.DiGraph
    ) -> List[Tuple[nx.DiGraph, float, str, str, float]]:
        """
        Find candidate motifs where target_node appears.
        Returns: List of (motif, score, matching_node, motif_type, demo_sim)
        """
        candidates = []
        target_label = current_graph.nodes[target_node].get("label", target_node)

        print(f"\nFinding candidates for node: {target_label}")

        if (
            not hasattr(self.motif_library, "semantic_motifs")
            or not self.motif_library.semantic_motifs
        ):
            print("Warning: Motif library is empty!")
            return candidates

        # Get demographic priorities
        priority_types = self._get_priority_motif_types()
        print(f"Priority types based on demographics: {priority_types}")

        # For each motif group in library
        for group_key, motifs in self.motif_library.semantic_motifs.items():

            for motif in motifs:
                try:
                    # Find nodes in motif that could match target node
                    matching_nodes = []
                    for node in motif.nodes():
                        node_label = motif.nodes[node].get("label", "")
                        sim = self.similarity_engine.node_similarity(
                            node_label, target_label
                        )
                        if sim >= self.node_merge_threshold:
                            matching_nodes.append(node)

                        if not matching_nodes:
                            continue

                    # Calculate various similarity scores
                    motif_type = motif.graph.get("motif_type", "Unknown")
                    type_weight = self._calculate_type_weight(
                        motif_type, priority_types
                    )

                    structural_score = self._calculate_motif_structural_fit(
                        motif, current_graph
                    )
                    coverage_score = self._calculate_coverage_score(
                        motif, covered_nodes
                    )
                    demo_sim = self._demographic_similarity(
                        motif, self.target_demographic
                    )

                    # Combined score with weights
                    final_score = (
                        0.3 * structural_score  # How well it fits structurally
                        + 0.3 * coverage_score  # How many new nodes it adds
                        + 0.2 * demo_sim  # Demographic match
                        + 0.2 * type_weight  # Priority based on motif type
                    )

                    # Add as candidate for each matching node
                    for node in matching_nodes:
                        candidates.append(
                            (motif, final_score, node, motif_type, demo_sim)
                        )

                except Exception as e:
                    print(f"Error processing motif: {e}")
                    continue

        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:10]  # Return top 10 candidates

    def reconstruct_graph(
        self, target_node: str, max_iterations: int = 100, min_score: float = 0.3
    ) -> nx.DiGraph:
        """
        Reconstruct graph using motif-based approach.
        Follows the algorithm:
        1. Start with seed node
        2. For each frontier node, find and integrate matching motifs
        3. Update frontier with upstream nodes
        4. Repeat until no frontier nodes or max iterations
        """
        print("Starting graph reconstruction...")
        G = nx.DiGraph()
        G.add_node(target_node, label=target_node)
        covered_nodes = {target_node}
        frontier = {target_node}

        for iteration in range(max_iterations):
            if not frontier:
                break

            print(f"\nIteration {iteration}")
            print(f"Frontier nodes: {frontier}")
            print(f"Current graph size: {len(G.nodes())} nodes")

            next_frontier = set()

            # Process each frontier node
            for node in frontier:
                # Find candidate motifs for this node
                candidates = self.find_motif_candidates_reverse(node, covered_nodes, G)

                # Take top 2 candidates above threshold
                selected = [c for c in candidates if c[1] >= min_score][:2]

                # Integrate selected motifs
                for motif, score, matching_node, motif_type, demo_sim in selected:
                    print(f"\nIntegrating motif (type={motif_type}, score={score:.3f})")

                    # Map the matching node to frontier node
                    node_mapping = self._integrate_motif_reverse(
                        G, motif, node, matching_node, covered_nodes
                    )

                    # Add upstream nodes to next frontier
                    upstream = set()
                    for n in node_mapping.values():
                        upstream.update(G.predecessors(n))
                    next_frontier.update(upstream - covered_nodes)

                    # If this is target node, remove any outgoing edges
                    if node == target_node:
                        out_edges = list(G.out_edges(node))
                        G.remove_edges_from(out_edges)

            frontier = next_frontier
            print(f"Next frontier size: {len(frontier)}")

        # Final cleanup
        self._clean_up_graph(G, target_node)
        return G

    def _clean_up_graph(self, G: nx.DiGraph, target_node: str) -> None:
        """
        Clean up the final graph
        """
        # 1. 移除 target_node 的出边
        outgoing = list(G.out_edges(target_node))
        for u, v in outgoing:
            G.remove_edge(u, v)

        # 2. 确保所有节点都有到 target_node 的路径
        for node in G.nodes():
            if node != target_node and not nx.has_path(G, node, target_node):
                G.add_edge(node, target_node, modifier=1.0)

        # 3. 移除重复的边
        edges_to_remove = []
        for u, v in G.edges():
            if G.has_edge(v, u):
                # 保留修饰符值较大的边
                uv_modifier = abs(G.edges[u, v].get("modifier", 1.0))
                vu_modifier = abs(G.edges[v, u].get("modifier", 1.0))
                if uv_modifier < vu_modifier:
                    edges_to_remove.append((u, v))
                else:
                    edges_to_remove.append((v, u))

        for edge in edges_to_remove:
            if G.has_edge(*edge):
                G.remove_edge(*edge)

    def _integrate_motif_reverse(
        self,
        G: nx.DiGraph,
        motif: nx.DiGraph,
        target_node: str,
        motif_node: str,
        covered_nodes: Set[str],
    ) -> Dict[str, str]:
        """
        Integrate a motif into the graph while ensuring path connectivity to target node.
        """
        # Initialize node mapping with the known match
        node_mapping = {motif_node: target_node}

        # First pass: Map all nodes
        for node in motif.nodes():
            if node in node_mapping:
                continue

            node_label = motif.nodes[node].get("label", "").replace(" ", "_").lower()

            # Try to find existing node to merge with
            best_match = None
            best_similarity = 0

            for existing_node in G.nodes():
                if existing_node == target_node:
                    continue
                existing_label = G.nodes[existing_node].get("label", "").lower()
                similarity = self.similarity_engine.node_similarity(
                    node_label, existing_label
                )

                if (
                    similarity > best_similarity
                    and similarity >= self.node_merge_threshold
                ):
                    best_similarity = similarity
                    best_match = existing_node

            # Create new node if no good match found
            if best_match:
                node_mapping[node] = best_match
            else:
                new_id = f"n{len(G.nodes()) + 1}"
                while new_id in G.nodes():
                    new_id = f"n{int(new_id[1:]) + 1}"

                G.add_node(new_id, label=node_label)
                node_mapping[node] = new_id

        # Second pass: Add edges and their attributes
        for u, v, data in motif.edges(data=True):
            mapped_u = node_mapping[u]
            mapped_v = node_mapping[v]

            # Skip edges from target node
            if mapped_u == target_node:
                continue

            # Skip if reverse edge exists
            if G.has_edge(mapped_v, mapped_u):
                continue

            # Add edge with attributes
            edge_attrs = data.copy()
            if "modifier" not in edge_attrs:
                edge_attrs["modifier"] = 1.0
            G.add_edge(mapped_u, mapped_v, **edge_attrs)

        # Ensure connectivity: add edges to connect any disconnected nodes to target_node
        for node in node_mapping.values():
            if node != target_node and not nx.has_path(G, node, target_node):
                # Add edge from disconnected node to target_node
                G.add_edge(node, target_node, modifier=1.0)

        # Update covered nodes
        covered_nodes.update(node_mapping.values())

        print(f"Integrated motif nodes: {list(node_mapping.values())}")

        return node_mapping

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
        Save the reconstructed graph in the new JSON format.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        graph_data = {
            "graph": {"nodes": {}, "edges": {}},
            "metadata": {"demographic": self.target_demographic},
        }

        # Add nodes
        for node in G.nodes():
            node_data = G.nodes[node]
            graph_data["graph"]["nodes"][node] = {
                "label": node_data.get("label", ""),
                "confidence": node_data.get("confidence", 0.0),
                "importance": node_data.get("importance", 0.0),
                "is_stance": node_data.get("is_stance", False),
            }

        # Add edges
        for i, (u, v) in enumerate(G.edges()):
            edge_data = G.edges[u, v]
            graph_data["graph"]["edges"][f"e{i}"] = {
                "source": u,
                "target": v,
                "modifier": edge_data.get("modifier", 1.0),
                "confidence": edge_data.get("confidence", 0.0),
            }

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


def main(args):
    """
    Example usage of the improved motif-based reconstruction.
    """
    print("\nInitializing reconstruction process...")
    print(f"Using motif library from: {args.motif_library}")

    # Check if file exists
    if not os.path.exists(args.motif_library):
        print(f"Error: Motif library file not found at {args.motif_library}")
        return

    # Use the proper load_library method
    try:
        library = MotifLibrary.load_library(args.motif_library)
        print(
            f"Successfully loaded motif library with {len(library.semantic_motifs)} motif groups"
        )
    except Exception as e:
        print(f"Error loading motif library: {e}")
        return

    print(f"\nMotif Library Statistics:")
    print(f"- Number of motif groups: {len(library.semantic_motifs)}")
    # Count total motifs across all groups
    total_motifs = sum(len(motifs) for motifs in library.semantic_motifs.values())
    print(f"- Total individual motifs: {total_motifs}")
    print(f"- Min motif size: {library.min_motif_size}")
    print(f"- Max motif size: {library.max_motif_size}")
    print(f"- Semantic similarity threshold: {library.min_semantic_similarity}")

    # Print motif type distribution
    motif_types = {}
    for group_key in library.semantic_motifs.keys():
        motif_type = group_key.split("_")[0]  # e.g., M1, M2.1, M3.1, M3.2
        motif_types[motif_type] = motif_types.get(motif_type, 0) + 1
    print("\nMotif Type Distribution:")
    for mtype, count in motif_types.items():
        print(f"- {mtype}: {count} groups")

    # Set target demographic
    print("\nTarget Demographic Information:")
    for key, value in args.target_demographic.items():
        print(f"- {key}: {value}")

    # Create reconstructor
    reconstructor = MotifBasedReconstructor(
        library,
        similarity_threshold=0.3,
        node_merge_threshold=0.8,
        target_demographic=args.target_demographic,
        demographic_weight=0.3,
    )

    print("\nStarting graph reconstruction...")
    # Reconstruct graph
    reconstructed_graph = reconstructor.reconstruct_graph(
        args.seed_node, max_iterations=args.max_iterations, min_score=0.3
    )

    # Analyze results
    print("\nReconstruction Results:")
    print(f"- Total nodes: {len(reconstructed_graph.nodes())}")
    print(f"- Total edges: {len(reconstructed_graph.edges())}")

    # Output node information
    print("\nNode Information:")
    for node in reconstructed_graph.nodes():
        label = reconstructed_graph.nodes[node].get("label", "")
        in_degree = reconstructed_graph.in_degree(node)
        out_degree = reconstructed_graph.out_degree(node)
        print(f"- {node}: {label} (in: {in_degree}, out: {out_degree})")

    # Create visualizations
    print("\nGenerating visualizations...")

    # Save as JSON
    json_path = reconstructor.save_as_json(
        reconstructed_graph, args.output_dir, "reconstructed_graph.json"
    )
    print(f"- JSON saved to: {json_path}")

    # Save as MMD
    mmd_path = reconstructor.save_as_mmd(
        reconstructed_graph, args.output_dir, "reconstructed_graph.mmd"
    )
    print(f"- MMD saved to: {mmd_path}")


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Motif-based graph reconstruction")

    # Add parameters
    parser.add_argument(
        "--seed_node",
        type=str,
        default="support_for_upzoning",
        help="Starting node for reconstruction",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=10, help="Maximum number of iterations"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(CURRENT_DIR, "graph_reconstruction"),
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--motif_library",
        type=str,
        default=os.path.join(CURRENT_DIR, "motif_library.json"),
        help="Path to motif library JSON file",
    )

    args = parser.parse_args()

    # Set example target demographic
    args.target_demographic = {
        "age": 28,
        "Geo Mobility": "Different house in United States 1 year ago",
        "householder type": "Renter",
        "Gross rent": "25.0 to 29.9 percent",
        "means of transportation": "Public Transportation",
        "income": "$50,000-$74,999",
        "occupation": "Management, business, science, and arts occupations",
        "marital status": "Never Married",
        "has children under 18": False,
        "children age range": "No Children",
        "ZIP code": 94105,
    }

    # args.target_demographic = {
    #     "age": 45,
    #     "Geo Mobility": "Same house 1 year ago",
    #     "householder type": "Owner-occupied",
    #     "Gross rent": "Not computed",
    #     "means of transportation": "Car / Truck / Van",
    #     "income": ">$200,000",
    #     "occupation": "Sales and office occupations",
    #     "marital status": "Now Married",
    #     "has children under 18": True,
    #     "children age range": "6 to 17 years old",
    #     "ZIP code": 92103,
    # }

    main(args)
