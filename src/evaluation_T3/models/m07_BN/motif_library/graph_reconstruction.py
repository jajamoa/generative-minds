"""
Motif-Based Causal Graph Reconstruction

This module implements an algorithm to reconstruct causal graphs using motifs as building blocks.
The reconstruction starts from a seed node (typically upzoning_stance) and grows the graph
by iteratively adding best-matching motifs.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Set, List, Tuple, Optional
from .motif_library import MotifLibrary, get_demographic_statistics
from .semantic_similarity import SemanticSimilarityEngine
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import argparse
import json


class MotifBasedReconstructor:
    def __init__(
        self,
        motif_library: MotifLibrary,
        similarity_threshold: float = 0.4,
        node_merge_threshold: float = 0.8,
        target_demographic: str = None,
        demographic_weight: float = 0.3,
    ):
        """
        Initialize the reconstructor.

        Args:
            motif_library: Loaded MotifLibrary instance
            similarity_threshold: Threshold for semantic similarity matching
            node_merge_threshold: Threshold for merging similar nodes
        """
        self.motif_library = motif_library
        self.similarity_threshold = similarity_threshold
        self.node_merge_threshold = node_merge_threshold
        self.similarity_engine = SemanticSimilarityEngine(use_wordnet=True)
        self.target_demographic = target_demographic
        self.demographic_weight = demographic_weight
        self.demographic_scores = self._calculate_demographic_scores()

    def _calculate_demographic_scores(self):
        """Calculate demographic similarity scores for all demographics."""
        scores = {}

        # Get demographic distribution
        total_samples = sum(self.motif_library.demographic_distribution.values())
        if total_samples == 0:
            return scores

        # Calculate score for each demographic
        for demo, count in self.motif_library.demographic_distribution.items():
            # Base score: inverse weight based on frequency (rare demographics score higher)
            rarity_score = 1.0 - (count / total_samples)

            # Similarity score: similarity to target demographic
            if self.target_demographic:
                similarity = self._demographic_similarity(demo, self.target_demographic)
            else:
                similarity = 0.5  # If no target demographic, assign medium score

            scores[demo] = 0.3 * rarity_score + 0.7 * similarity

        return scores

    def _demographic_similarity(self, demo_of_motif: str, demo_of_target: str) -> float:
        """Calculate similarity between two demographics."""
        NONSENSE_DEMOGRAPHICS = ["unknown", "other", "", " ", "none", None]
        if (
            demo_of_target in NONSENSE_DEMOGRAPHICS
            or demo_of_motif in NONSENSE_DEMOGRAPHICS
        ):
            return 0.0

        if demo_of_motif == demo_of_target:
            return 1.0

        # Simple similarity calculation, can be customized based on actual demographic categories
        # For example, adjacent age groups have higher similarity
        demographic_map = {
            "young": ["middle_aged"],
            "middle_aged": ["young", "elderly"],
            "elderly": ["middle_aged"],
            "urban": ["suburban"],
            "suburban": ["urban", "rural"],
            "rural": ["suburban"],
        }

        if (
            demo_of_motif in demographic_map
            and demo_of_target in demographic_map[demo_of_motif]
        ):
            return 0.6

        return 0.2  # Default low similarity

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

        # Search through all motif groups in library
        for group_key, motifs in self.motif_library.semantic_motifs.items():
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

                        # Combined score with higher weight on semantic similarity
                        score = 0.6 * similarity + 0.4 * extendability
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

    def reconstruct_graph(
        self, target_node: str, max_iterations: int = 100, min_score: float = 0.3
    ) -> nx.DiGraph:
        """
        Reconstruct causal graph with improved diversity and node merging.
        Also tracks demographic diversity.
        """
        G = nx.DiGraph()
        G.add_node(target_node, label=target_node)

        covered_nodes = {target_node}
        frontier = {target_node}
        used_motif_types = set()

        # Initialize demographic tracking
        self.used_demographics = set()

        iteration = 0
        while frontier and iteration < max_iterations:
            iteration += 1
            new_frontier = set()

            for f_node in frontier:
                # Now candidates returns (motif, score, motif_node, group_key, demographic)
                candidates = self.find_motif_candidates_reverse(
                    f_node, covered_nodes, G
                )

                # Updated unpacking to include demographic - 这里是修复的地方
                for motif, score, motif_node, group_key, demographic in candidates:
                    if score >= min_score:
                        motif_type = group_key.split("_")[0]

                        # Consider both motif type and demographic diversity
                        if (
                            motif_type not in used_motif_types
                            or demographic not in self.used_demographics
                            or len(used_motif_types) >= 3
                        ):
                            self._integrate_motif_reverse(
                                G, motif, f_node, covered_nodes
                            )
                            used_motif_types.add(motif_type)

                            # Update frontier with new nodes
                            new_nodes = set(G.nodes()) - covered_nodes
                            new_frontier.update(new_nodes)

                            # Only break inner loop if we added a motif
                            break

                frontier = new_frontier

                # Early stopping if graph becomes too large
                if len(G.nodes()) > 50:
                    break

        return G

    def find_motif_candidates_reverse(
        self, target_node: str, covered_nodes: Set[str], current_graph: nx.DiGraph
    ) -> List[Tuple[nx.DiGraph, float, str, str, str]]:
        """
        Find candidate motifs where target node is the outcome, with demographic scoring.
        Enhanced to promote motif diversity and better coverage.

        Returns:
            List of tuples: (motif, score, motif_node, group_key, demographic)
        """
        candidates = []
        target_label = current_graph.nodes[target_node].get("label", target_node)
        used_motif_types = set()  # Track used motif types

        for group_key, motifs in self.motif_library.semantic_motifs.items():
            motif_type = group_key.split("_")[
                0
            ]  # Extract base motif type (M1, M2.3, etc.)

            for motif in motifs:
                # Find outcome nodes (nodes with no outgoing edges)
                for motif_node in motif.nodes():
                    if motif.out_degree(motif_node) == 0:
                        motif_label = motif.nodes[motif_node].get("label", "")

                        # Calculate comprehensive score
                        semantic_sim = self.similarity_engine.node_similarity(
                            motif_label, target_label
                        )

                        if semantic_sim >= self.similarity_threshold:
                            # Calculate additional scores
                            structural_score = self._calculate_structural_fit(
                                motif, motif_node, current_graph, target_node
                            )
                            coverage_score = self._calculate_coverage_score(
                                motif, covered_nodes
                            )

                            # Add diversity bonus if motif type not used
                            diversity_bonus = (
                                0.2 if motif_type not in used_motif_types else 0
                            )

                            # Get motif's demographic information
                            motif_demographic = motif.graph.get(
                                "demographic", "unknown"
                            )

                            # Calculate demographic score
                            demo_score = self.demographic_scores.get(
                                motif_demographic, 0.5
                            )

                            # Add demographic diversity bonus
                            demo_diversity_bonus = 0.0
                            if hasattr(self, "used_demographics"):
                                if motif_demographic not in self.used_demographics:
                                    demo_diversity_bonus = 0.1

                            # Weighted combination of scores with demographic consideration
                            final_score = (1 - self.demographic_weight) * (
                                0.4 * semantic_sim
                                + 0.2 * structural_score
                                + 0.2 * coverage_score
                                + 0.2 * diversity_bonus
                            ) + self.demographic_weight * (
                                0.8 * demo_score + 0.2 * demo_diversity_bonus
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
                            used_motif_types.add(motif_type)

        # Sort by score but ensure diversity in top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Filter to ensure diverse motif types in top results
        diverse_candidates = []
        seen_types = set()
        seen_demographics = set()

        for candidate in candidates:
            motif_type = candidate[3].split("_")[0]
            demographic = candidate[4]

            # Prioritize diversity in both motif type and demographic
            if motif_type not in seen_types or demographic not in seen_demographics:
                diverse_candidates.append(candidate)
                seen_types.add(motif_type)
                seen_demographics.add(demographic)
                if len(diverse_candidates) >= 5:  # Limit to top 5 diverse motifs
                    break

        # If we don't have enough diverse candidates, add the highest scoring ones
        if len(diverse_candidates) < 5:
            for candidate in candidates:
                if candidate not in diverse_candidates:
                    diverse_candidates.append(candidate)
                    if len(diverse_candidates) >= 5:
                        break

        return diverse_candidates

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
        covered_nodes: Set[str],
    ) -> None:
        """
        Integrate a motif into the graph with improved node merging and relationship preservation.
        """
        node_mapping = {}
        node_mapping[target_node] = target_node

        # Track node labels for duplicate detection
        label_to_node = {G.nodes[n].get("label", ""): n for n in G.nodes()}

        # First add all nodes with improved merging
        for node in motif.nodes():
            if node in node_mapping:
                continue

            new_label = motif.nodes[node].get("label", "")

            # First check exact label matches
            if new_label in label_to_node:
                node_mapping[node] = label_to_node[new_label]
                continue

            # Then check semantic similarity
            mergeable_node = self._find_mergeable_node(G, new_label)

            if mergeable_node:
                node_mapping[node] = mergeable_node
                label_to_node[new_label] = mergeable_node
            else:
                # Generate unique node ID
                node_id = f"n{len(G.nodes()) + 1}"
                G.add_node(node_id, **motif.nodes[node])
                node_mapping[node] = node_id
                label_to_node[new_label] = node_id

        # Then add edges with preserved attributes
        for u, v in motif.edges():
            mapped_u = node_mapping[u]
            mapped_v = node_mapping[v]
            if not G.has_edge(mapped_u, mapped_v):
                # Get edge attributes from motif
                edge_attrs = motif.edges[u, v]
                # Ensure modifier exists
                if "modifier" not in edge_attrs:
                    edge_attrs["modifier"] = 1.0  # Default to positive influence
                # Add edge with attributes
                G.add_edge(mapped_u, mapped_v, **edge_attrs)

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
        default=os.path.join(CURRENT_DIR, "motif_library.json"),
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
