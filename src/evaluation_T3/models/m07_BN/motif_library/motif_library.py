"""
Motif Library - Create, manage and augment motifs from causal graphs.

This module implements a two-step process to identify motifs in cognitive causal graphs:
1. Topology-Based Candidate Grouping: Using graph isomorphism to identify structural patterns
2. Semantic Filtering: Using semantic similarity to group functionally equivalent patterns

The library also provides methods for data augmentation using nearest neighbor weighting
and bootstrapping techniques.
"""

import os
import json
import pickle
import networkx as nx
import numpy as np
from collections import defaultdict
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict
import itertools
import pdb  # 添加在文件开头的 imports 部分


from semantic_similarity import SemanticSimilarityEngine


def ensure_evaluation_prefix(path: str) -> str:
    """Ensure path has src/evaluation prefix if not already present."""
    prefix = "src/evaluation_T3"
    if not path.startswith(prefix) and not path.startswith("/"):
        return os.path.join(prefix, path)
    return path


responses_file_path = ensure_evaluation_prefix(
    "experiment/eval/data/sf_prolific_survey/causal_graph_responses_5.11_with_geo.json"
)


class MotifLibrary:
    """
    A library for storing, managing, and augmenting motifs extracted from cognitive causal graphs.

    Provides functionality for:
    - Two-stage motif extraction (topology + semantic filtering)
    - Motif visualization and analysis
    - Data augmentation using nearest neighbor and bootstrapping
    - Saving/loading library for persistence
    """

    def __init__(self, min_motif_size=3, max_motif_size=5, min_semantic_similarity=0.4):
        """
        Initialize the motif library.

        Args:
            min_motif_size: Minimum number of nodes in a motif
            max_motif_size: Maximum number of nodes in a motif
            min_semantic_similarity: Minimum semantic similarity threshold (0-1)
        """
        self.min_motif_size = min_motif_size
        self.max_motif_size = max_motif_size
        self.min_semantic_similarity = min_semantic_similarity

        # Initialize motif storage structures
        self.topological_motifs = {}  # Dictionary keyed by motif type and size
        self.semantic_motifs = {}  # Dictionary keyed by semantic group ID
        self.motif_metadata = {}  # Store metadata for each motif group
        self.motif_demographics = {}  # Store demographic info for each motif
        self.demographic_distribution = {}  # Overall demographic distribution

        # Initialize semantic similarity engine
        self.similarity_engine = SemanticSimilarityEngine(use_wordnet=True)

        # Define basic motif templates by type
        self.motif_types = {
            "M1": "Chain",  # A → B → C
            "M2.1": "Basic Fork",  # A → B, A → C (1-to-2)
            "M2.2": "Extended Fork",  # A → B, A → C, A → D (1-to-3)
            "M2.3": "Large Fork",  # A → B, A → C, A → D, A → E, ... (1-to-4+)
            "M3.1": "Basic Collider",  # A → C, B → C (2-to-1)
            "M3.2": "Extended Collider",  # A → D, B → D, C → D (3-to-1)
            "M3.3": "Large Collider",  # A → E, B → E, C → E, D → E, ... (4+-to-1)
        }

        # Initialize library stats
        self.stats = {
            "total_topological_motifs": 0,
            "total_semantic_groups": 0,
            "total_augmented_motifs": 0,
        }

    def create_motif_template(self, motif_type, size=3):
        """
        Create a template graph for a specific motif type.

        Args:
            motif_type: Type of motif (e.g., 'M1', 'M2.1')
            size: Size of the motif (nodes)

        Returns:
            NetworkX DiGraph representing the motif template
        """
        G = nx.DiGraph()

        if motif_type == "M1":  # Chain
            # Create a simple path of 'size' nodes
            for i in range(size - 1):
                G.add_edge(f"n{i}", f"n{i+1}")

        elif motif_type == "M2.1":  # Basic Fork (1-to-2)
            G.add_edge("n0", "n1")
            G.add_edge("n0", "n2")
            # Add additional nodes in chain if size > 3
            for i in range(3, size):
                # Connect to the second branch
                G.add_edge(f"n2", f"n{i}")

        elif motif_type == "M2.2":  # Extended Fork (1-to-3)
            G.add_edge("n0", "n1")
            G.add_edge("n0", "n2")
            G.add_edge("n0", "n3")
            # Add additional nodes in chain if size > 4
            for i in range(4, size):
                # Connect to the third branch
                G.add_edge(f"n3", f"n{i}")

        elif motif_type == "M2.3":  # Large Fork (1-to-4+)
            # Create a star with a center node connecting to all others
            for i in range(1, size):
                G.add_edge("n0", f"n{i}")

        elif motif_type == "M3.1":  # Basic Collider (2-to-1)
            G.add_edge("n0", "n2")
            G.add_edge("n1", "n2")
            # Add additional nodes in chain if size > 3
            for i in range(3, size):
                # Connect from the first source
                G.add_edge(f"n{i-1}", f"n{i}")

        elif motif_type == "M3.2":  # Extended Collider (3-to-1)
            G.add_edge("n0", "n3")
            G.add_edge("n1", "n3")
            G.add_edge("n2", "n3")
            # Add additional nodes in chain if size > 4
            for i in range(4, size):
                # Connect from the sink
                G.add_edge(f"n3", f"n{i}")

        elif motif_type == "M3.3":  # Large Collider (4+-to-1)
            # Create a reversed star with all nodes connecting to one
            sink_node = f"n{size-1}"
            for i in range(size - 1):
                G.add_edge(f"n{i}", sink_node)

        # Add dummy labels to nodes
        for node in G.nodes():
            G.nodes[node]["label"] = f"node_{node}"

        return G

    def get_demographic_statistics(samples_dir: str) -> dict:
        demographic_stats = {}
        total_samples = 0

        for filename in os.listdir(samples_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(samples_dir, filename)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    metadata = data.get("metadata", {})
                    demographic = metadata.get("perspective", "unknown")

                    if demographic not in demographic_stats:
                        demographic_stats[demographic] = {"count": 0, "samples": []}

                    demographic_stats[demographic]["count"] += 1
                    demographic_stats[demographic]["samples"].append(filename)
                    total_samples += 1

                except Exception as e:
                    print(f"Error reading {filename}: {e}")

        for demo in demographic_stats:
            demographic_stats[demo]["percentage"] = (
                demographic_stats[demo]["count"] / total_samples * 100
            )

        return {
            "distribution": demographic_stats,
            "total_samples": total_samples,
            "unique_demographics": list(demographic_stats.keys()),
        }

    def extract_topological_motifs(
        self, G, sample_id=None, demographic_info=None, motif_types=None
    ):
        """
        Extract motifs from a graph using topology-based analysis with demographic tracking.
        Ensures each motif is a minimal unit matching M1, M2.x, or M3.x patterns.
        """
        motifs = []

        # Ensure graph is not empty
        if G is None or G.number_of_nodes() == 0:
            return motifs

        # Create a copy of the graph to modify
        working_graph = G.copy()

        # Remove outgoing edges from stance nodes
        stance_nodes = [
            node
            for node in working_graph.nodes()
            if working_graph.nodes[node].get("is_stance", False)
        ]
        edges_to_remove = []
        for node in stance_nodes:
            edges_to_remove.extend(
                [(node, succ) for succ in working_graph.successors(node)]
            )

        working_graph.remove_edges_from(edges_to_remove)
        if edges_to_remove:
            print(
                f"Removed {len(edges_to_remove)} outgoing edges from {len(stance_nodes)} stance nodes"
            )

        # Get node labels mapping
        node_labels = nx.get_node_attributes(working_graph, "label")

        def create_motif_with_metadata(nodes, motif_type=None):
            """Helper function to create a motif with metadata"""
            subgraph = working_graph.subgraph(nodes)
            motif = nx.DiGraph(subgraph)

            # Copy node attributes
            for n in nodes:
                motif.nodes[n]["label"] = node_labels.get(n, "")
                motif.nodes[n]["is_stance"] = working_graph.nodes[n].get(
                    "is_stance", False
                )
                motif.nodes[n]["confidence"] = working_graph.nodes[n].get(
                    "confidence", 0
                )
                motif.nodes[n]["importance"] = working_graph.nodes[n].get(
                    "importance", 0
                )

            # Add demographics and motif type
            motif.graph["metadata"] = (
                [
                    {
                        "agent_id": sample_id,
                        "demographic": demographic_info.get("agent", {}),
                    }
                ]
                if demographic_info.get("agent", {})
                else []
            )

            if motif_type:
                motif.graph["motif_type"] = motif_type

            return motif

        def is_valid_chain(nodes):
            """Check if nodes form a valid chain pattern"""
            if len(nodes) != 3:
                return False
            subgraph = working_graph.subgraph(nodes)
            if subgraph.number_of_edges() != 2:
                return False
            # Must have exactly one source and one sink
            in_degrees = dict(subgraph.in_degree())
            out_degrees = dict(subgraph.out_degree())
            return (
                sum(1 for d in in_degrees.values() if d == 0) == 1
                and sum(1 for d in out_degrees.values() if d == 0) == 1
            )

        def is_valid_fork(source, targets):
            """Check if nodes form a valid fork pattern"""
            subgraph = working_graph.subgraph([source] + list(targets))
            # Source should only have outgoing edges to targets
            return all(working_graph.has_edge(source, t) for t in targets) and all(
                working_graph.in_degree(t) == 1 for t in targets
            )

        def is_valid_collider(sink, sources):
            """Check if nodes form a valid collider pattern"""
            subgraph = working_graph.subgraph([sink] + list(sources))
            # Sink should only have incoming edges from sources
            return all(working_graph.has_edge(s, sink) for s in sources) and all(
                working_graph.out_degree(s) == 1 for s in sources
            )

        # Extract Chain patterns (M1)
        for node in working_graph.nodes():
            # Look for chains starting from this node
            for succ1 in working_graph.successors(node):
                for succ2 in working_graph.successors(succ1):
                    chain_nodes = [node, succ1, succ2]
                    if is_valid_chain(chain_nodes):
                        motifs.append(create_motif_with_metadata(chain_nodes, "M1"))

        # Extract Fork patterns (M2.x)
        for node in working_graph.nodes():
            successors = list(working_graph.successors(node))
            out_degree = len(successors)
            if out_degree >= 2:
                # For each possible fork size (2 to 4+)
                for size in range(2, min(out_degree + 1, 5)):
                    for targets in itertools.combinations(successors, size):
                        if is_valid_fork(node, targets):
                            fork_type = f"M2.{min(size-1, 3)}"  # M2.1, M2.2, or M2.3
                            fork_nodes = [node] + list(targets)
                            motifs.append(
                                create_motif_with_metadata(fork_nodes, fork_type)
                            )

        # Extract Collider patterns (M3.x)
        for node in working_graph.nodes():
            predecessors = list(working_graph.predecessors(node))
            in_degree = len(predecessors)
            if in_degree >= 2:
                # For each possible collider size (2 to 4+)
                for size in range(2, min(in_degree + 1, 5)):
                    for sources in itertools.combinations(predecessors, size):
                        if is_valid_collider(node, sources):
                            collider_type = (
                                f"M3.{min(size-1, 3)}"  # M3.1, M3.2, or M3.3
                            )
                            collider_nodes = list(sources) + [node]
                            motifs.append(
                                create_motif_with_metadata(
                                    collider_nodes, collider_type
                                )
                            )

        # Remove any overlapping or invalid motifs
        final_motifs = []
        seen_node_sets = set()

        for motif in motifs:
            node_set = frozenset(motif.nodes())
            if node_set not in seen_node_sets:
                # Verify the motif is minimal
                if len(motif.nodes()) <= 4:  # Maximum size for our defined patterns
                    final_motifs.append(motif)
                    seen_node_sets.add(node_set)

        return final_motifs

    def compute_node_position_info(self, motif):
        """
        Compute structural position information for each node
        """
        positions = {}
        for node in motif.nodes():
            in_deg = motif.in_degree(node)
            out_deg = motif.out_degree(node)

            if in_deg == 0:
                positions[node] = "source"
            elif out_deg == 0:
                positions[node] = "sink"
            else:
                positions[node] = "intermediate"

        return positions

    def compute_enhanced_similarity(self, motif1, motif2, node_mapping):
        """
        Compute enhanced semantic similarity between motifs with position weighting
        using the formula from the paper:
        Sim(m1, m2) = Σ(w_i * s_i) / Σ(w_i)
        """
        if not node_mapping:
            return 0.0

        # Position weights as specified in the paper
        weights = {"source": 1.5, "sink": 1.5, "intermediate": 1.0}

        # Get node positions
        positions1 = self.compute_node_position_info(motif1)
        positions2 = self.compute_node_position_info(motif2)

        total_weighted_sim = 0.0
        total_weight = 0.0

        for node1, node2 in node_mapping.items():
            # Get node labels
            label1 = motif1.nodes[node1].get("label", str(node1))
            label2 = motif2.nodes[node2].get("label", str(node2))

            # Calculate semantic similarity
            sim = self.similarity_engine.node_similarity(label1, label2)

            # Get position weights
            pos1 = positions1[node1]
            pos2 = positions2[node2]
            weight = (weights[pos1] + weights[pos2]) / 2

            total_weighted_sim += weight * sim
            total_weight += weight

        return total_weighted_sim / total_weight if total_weight > 0 else 0.0

    def apply_semantic_filtering(self, topological_motifs=None):
        """
        Apply semantic filtering and augmentation to build a robust motif library.
        """
        if topological_motifs is None:
            topological_motifs = self.topological_motifs

        semantic_groups = {}
        stats = {
            "total_input_motifs": 0,
            "total_output_motifs": 0,
            "total_groups": 0,
            "motifs_per_type": defaultdict(int),
            "groups_per_type": defaultdict(int),
            "augmented_motifs": 0,
            # Add new statistics for node merging
            "pre_merge_nodes": 0,
            "post_merge_nodes": 0,
            "merged_nodes_by_type": defaultdict(lambda: {"before": 0, "after": 0}),
        }

        # First pass: Group by motif type
        typed_motifs = defaultdict(list)
        for key, motifs in topological_motifs.items():
            for motif in motifs:
                motif_type = self.identify_motif_type(motif)
                typed_motifs[motif_type].append(motif)
                stats["motifs_per_type"][motif_type] += 1
                stats["total_input_motifs"] += 1
                # Count initial nodes
                stats["pre_merge_nodes"] += len(motif.nodes())
                stats["merged_nodes_by_type"][motif_type]["before"] += len(
                    motif.nodes()
                )

        print("\nApplying semantic filtering and augmentation...")

        # Process each motif type separately
        for motif_type, motifs in typed_motifs.items():
            print(f"\nProcessing {motif_type} motifs ({len(motifs)} instances)...")
            pre_merge_nodes = sum(len(m.nodes()) for m in motifs)
            print(f"Pre-merge nodes: {pre_merge_nodes}")

            # Skip if only one motif
            if len(motifs) <= 1:
                group_key = f"{motif_type}_1"
                semantic_groups[group_key] = {
                    "instances": motifs,
                    "abstract_motif": motifs[0],
                    "demographics": self.get_group_demographics(motifs),
                }
                stats["groups_per_type"][motif_type] += 1
                stats["post_merge_nodes"] += pre_merge_nodes
                stats["merged_nodes_by_type"][motif_type]["after"] += pre_merge_nodes
                continue

            # Group similar motifs
            processed = set()
            group_id = 1
            post_merge_nodes = 0

            for i, motif1 in enumerate(motifs):
                if i in processed:
                    continue

                # Create new group
                group_key = f"{motif_type}_{group_id}"
                current_group = {
                    "instances": [motif1],
                    "abstract_motif": motif1.copy(),
                    "demographics": self.get_group_demographics([motif1]),
                }
                processed.add(i)
                merged_in_group = len(motif1.nodes())

                # Find similar motifs
                for j, motif2 in enumerate(motifs):
                    if j in processed or i == j:
                        continue

                    # Check isomorphism and semantic similarity
                    matcher = nx.algorithms.isomorphism.DiGraphMatcher(motif1, motif2)
                    if matcher.is_isomorphic():
                        mapping = next(matcher.isomorphisms_iter())
                        node_sim = self.compute_enhanced_similarity(
                            motif1, motif2, mapping
                        )
                        edge_sim = self.compute_edge_similarity(motif1, motif2, mapping)

                        combined_sim = 0.7 * node_sim + 0.3 * edge_sim

                        if combined_sim >= self.min_semantic_similarity:
                            merged_motif = self.merge_motifs(
                                current_group["abstract_motif"], motif2, mapping
                            )
                            current_group["abstract_motif"] = merged_motif
                            current_group["instances"].append(motif2)
                            current_group["demographics"] = self.get_group_demographics(
                                current_group["instances"]
                            )
                            processed.add(j)

                post_merge_nodes += merged_in_group

                # Apply data augmentation
                augmented_motifs = self.augment_group(current_group)
                current_group["instances"].extend(augmented_motifs)
                stats["augmented_motifs"] += len(augmented_motifs)

                # Add metadata
                self.motif_metadata[group_key] = {
                    "motif_type": motif_type,
                    "description": self.get_motif_description(motif_type),
                    "size": len(current_group["abstract_motif"].nodes()),
                    "edges": len(current_group["abstract_motif"].edges()),
                    "instances": len(current_group["instances"]),
                    "demographics": current_group["demographics"],
                    # Add merging statistics to metadata
                    "pre_merge_nodes": pre_merge_nodes,
                    "post_merge_nodes": post_merge_nodes,
                }

                semantic_groups[group_key] = current_group
                group_id += 1
                stats["groups_per_type"][motif_type] += 1

            stats["post_merge_nodes"] += post_merge_nodes
            stats["merged_nodes_by_type"][motif_type]["after"] += post_merge_nodes
            print(f"Post-merge nodes: {post_merge_nodes}")
            print(
                f"Node reduction: {((pre_merge_nodes - post_merge_nodes) / pre_merge_nodes * 100):.1f}%"
            )

        # Update statistics
        stats["total_output_motifs"] = sum(
            len(group["instances"]) for group in semantic_groups.values()
        )
        stats["total_groups"] = len(semantic_groups)

        # Print detailed statistics
        self._print_filtering_stats(stats)
        print("\nNode Merging Statistics:")
        print(f"Total nodes before merging: {stats['pre_merge_nodes']}")
        print(f"Total nodes after merging: {stats['post_merge_nodes']}")
        print(
            f"Node reduction: {((stats['pre_merge_nodes'] - stats['post_merge_nodes']) / stats['pre_merge_nodes'] * 100):.1f}%"
        )
        print("\nBy motif type:")
        for mtype in sorted(stats["merged_nodes_by_type"].keys()):
            before = stats["merged_nodes_by_type"][mtype]["before"]
            after = stats["merged_nodes_by_type"][mtype]["after"]
            if before > 0:
                reduction = (before - after) / before * 100
                print(
                    f"  {mtype}: {before} → {after} nodes ({reduction:.1f}% reduction)"
                )

        # Update library
        self.semantic_motifs = semantic_groups
        self.stats["total_semantic_groups"] = len(semantic_groups)
        self.stats["filtering_stats"] = stats

        return semantic_groups

    def compute_edge_similarity(self, motif1, motif2, node_mapping):
        """
        Compute similarity between edge attributes of two motifs
        """
        if not node_mapping:
            return 0.0

        total_sim = 0.0
        edge_count = 0

        for node1, node2 in node_mapping.items():
            for succ1 in motif1.successors(node1):
                if succ1 in node_mapping:
                    succ2 = node_mapping[succ1]
                    if motif2.has_edge(node2, succ2):
                        # Compare edge attributes
                        edge1 = motif1.edges[node1, succ1]
                        edge2 = motif2.edges[node2, succ2]

                        # Calculate attribute similarities
                        modifier_sim = (
                            1.0
                            - abs(edge1.get("modifier", 0) - edge2.get("modifier", 0))
                            / 2.0
                        )
                        conf_sim = 1.0 - abs(
                            edge1.get("confidence", 0) - edge2.get("confidence", 0)
                        )

                        # Weighted combination
                        edge_sim = 0.6 * modifier_sim + 0.4 * conf_sim
                        total_sim += edge_sim
                        edge_count += 1

        return total_sim / edge_count if edge_count > 0 else 0.0

    def update_abstract_motif(self, abstract_motif, new_motif, mapping):
        """Update abstract motif with merged information from new motif"""
        # Existing label and confidence merging logic
        for node1, node2 in mapping.items():
            # Update node labels and confidence as before
            if (
                "label" in abstract_motif.nodes[node1]
                and "label" in new_motif.nodes[node2]
            ):
                # ... existing label merging code ...
                pass

        # Merge demographics
        abstract_demos = abstract_motif.graph.get("demographics", [])
        new_demos = new_motif.graph.get("demographics", [])

        # Create a dictionary of demographics keyed by agent_id
        demo_dict = {demo["agent_id"]: demo for demo in abstract_demos}

        # Add new demographics if agent_id not already present
        for demo in new_demos:
            agent_id = demo["agent_id"]
            if agent_id not in demo_dict:
                demo_dict[agent_id] = demo

        # Update the abstract motif with merged demographics
        abstract_motif.graph["demographics"] = list(demo_dict.values())

    def augment_group(self, group):
        """
        Apply data augmentation strategies to a motif group
        """
        augmented_motifs = []

        # 1. Demographic Interpolation
        demographics = group["demographics"]
        if len(demographics) >= 2:
            interpolated = self.demographic_interpolation(
                group["instances"], demographics
            )
            augmented_motifs.extend(interpolated)

        # 2. Perturbative Bootstrap
        if len(group["instances"]) >= 3:
            bootstrapped = self.perturbative_bootstrap(group["instances"])
            augmented_motifs.extend(bootstrapped)

        return augmented_motifs

    def demographic_interpolation(self, motifs, demographics, num_samples=2):
        """
        Generate new motifs by interpolating between similar demographic profiles
        """
        interpolated = []
        for i in range(num_samples):
            # Select two random motifs
            m1, m2 = random.sample(motifs, 2)
            d1, d2 = random.sample(demographics, 2)

            # Create interpolated motif
            new_motif = nx.DiGraph()
            # ... implement interpolation logic based on demographic distances ...

            interpolated.append(new_motif)
        return interpolated

    def perturbative_bootstrap(self, motifs, num_samples=2):
        """
        Generate synthetic motifs using bootstrap resampling with perturbations
        """
        bootstrapped = []
        for i in range(num_samples):
            # Select a random motif
            base_motif = random.choice(motifs)

            # Create perturbed copy
            new_motif = base_motif.copy()

            # Add Gaussian noise to numerical attributes
            for node in new_motif.nodes():
                for attr in ["confidence", "importance"]:
                    val = new_motif.nodes[node].get(attr, 0)
                    noise = np.random.normal(0, 0.1)  # 10% noise
                    new_motif.nodes[node][attr] = max(0, min(1, val + noise))

            bootstrapped.append(new_motif)
        return bootstrapped

    def _print_filtering_stats(self, stats):
        """Print detailed statistics about the filtering process"""
        print("\nMotif Filtering Statistics:")
        print(f"Total input motifs: {stats['total_input_motifs']}")

        # Calculate motifs after filtering but before augmentation
        filtered_motifs = stats["total_output_motifs"] - stats["augmented_motifs"]
        print(f"Motifs after filtering: {filtered_motifs}")
        print(f"Augmented motifs added: {stats['augmented_motifs']}")
        print(f"Total output motifs: {stats['total_output_motifs']}")
        print(f"Total semantic groups: {stats['total_groups']}")

        print("\nMotifs by type:")
        for mtype in sorted(stats["motifs_per_type"].keys()):
            print(
                f"  {mtype}: {stats['motifs_per_type'][mtype]} motifs, "
                f"{stats['groups_per_type'][mtype]} groups"
            )

        if stats["total_input_motifs"] > 0:
            # Calculate reduction from filtering
            filtering_reduction = (
                1 - filtered_motifs / stats["total_input_motifs"]
            ) * 100
            # Calculate expansion from augmentation
            augmentation_expansion = (
                (stats["total_output_motifs"] / filtered_motifs - 1) * 100
                if filtered_motifs > 0
                else 0
            )

            print(
                f"\nReduction from filtering: {filtering_reduction:.1f}% "
                f"({stats['total_input_motifs']} → {filtered_motifs} motifs)"
            )
            print(
                f"Expansion from augmentation: +{augmentation_expansion:.1f}% "
                f"({filtered_motifs} → {stats['total_output_motifs']} motifs)"
            )

    def get_motif_description(self, motif_type):
        """Get human-readable description of motif type"""
        descriptions = {
            "M1": "Chain",
            "M2.1": "Basic Fork",
            "M2.2": "Extended Fork",
            "M2.3": "Large Fork",
            "M3.1": "Basic Collider",
            "M3.2": "Extended Collider",
            "M3.3": "Large Collider",
        }
        return descriptions.get(motif_type, "Unknown Type")

    def get_group_demographics(self, motifs):
        """Extract and merge demographic information for a group of motifs"""
        demographics = []
        seen_agent_ids = set()

        for motif in motifs:
            if "demographics" in motif.graph:
                for demo in motif.graph["demographics"]:
                    agent_id = demo["agent_id"]
                    if agent_id not in seen_agent_ids:
                        demographics.append(demo)
                        seen_agent_ids.add(agent_id)
        return demographics

    def calculate_motif_vector(self, G):
        """
        Calculate the motif frequency vector for a graph.

        Args:
            G: NetworkX DiGraph to analyze

        Returns:
            Dictionary mapping motif groups to their frequency
        """
        # Extract topological motifs from the graph
        graph_motifs = self.extract_topological_motifs(G)

        # Filter motifs semantically
        graph_semantic_motifs = self.apply_semantic_filtering(graph_motifs)

        # Calculate motif frequencies
        total_motifs = sum(len(motifs) for motifs in graph_semantic_motifs.values())
        if total_motifs == 0:
            return {}

        motif_vector = {
            group_key: len(motifs) / total_motifs
            for group_key, motifs in graph_semantic_motifs.items()
        }

        return motif_vector

    def augment_by_nearest_neighbor(self, num_samples=10):
        """
        Augment the motif library using a nearest neighbor weighted approach.

        Generates new synthetic motifs by combining features of similar motifs.

        Args:
            num_samples: Number of augmented samples to generate per motif group

        Returns:
            Dictionary of augmented motifs
        """
        if not self.semantic_motifs:
            print("Error: No motifs in the library to augment. Extract motifs first.")
            return {}

        print(
            f"\nPerforming nearest neighbor augmentation ({num_samples} samples per group)..."
        )
        augmented_motifs = {}

        # Process each semantic group
        for group_key, motifs in tqdm(
            self.semantic_motifs.items(), desc="Augmenting groups"
        ):
            if len(motifs) < 2:
                # Skip groups with only one motif
                continue

            # Create new synthetic motifs
            new_motifs = []

            for _ in range(num_samples):
                # Randomly select a base motif
                base_motif = random.choice(motifs)

                # Create a copy of the base motif
                synthetic_motif = base_motif.copy()

                # Randomly select a similar motif to borrow from
                similar_motif = random.choice([m for m in motifs if m != base_motif])

                # Find isomorphism mapping between the motifs
                matcher = nx.algorithms.isomorphism.DiGraphMatcher(
                    base_motif, similar_motif
                )
                if matcher.is_isomorphic():
                    mapping = next(matcher.isomorphisms_iter())

                    # With 50% probability, borrow node labels from the similar motif
                    if random.random() < 0.5:
                        for node1, node2 in mapping.items():
                            if random.random() < 0.3:  # Only modify some nodes
                                label1 = base_motif.nodes[node1].get("label", "")
                                label2 = similar_motif.nodes[node2].get("label", "")

                                # Blend the labels
                                if label1 and label2:
                                    words1 = label1.split()
                                    words2 = label2.split()
                                    if len(words1) > 1 and len(words2) > 1:
                                        # Create hybrid label by mixing words
                                        new_label = " ".join(
                                            random.sample(
                                                words1, min(len(words1) // 2, 1)
                                            )
                                            + random.sample(
                                                words2, min(len(words2) // 2, 1)
                                            )
                                        )
                                        synthetic_motif.nodes[node1][
                                            "label"
                                        ] = new_label

                new_motifs.append(synthetic_motif)

            # Add the augmented motifs to the library
            aug_key = f"{group_key}_augmented"
            augmented_motifs[aug_key] = new_motifs

            # Add metadata
            self.motif_metadata[aug_key] = {
                **self.motif_metadata.get(group_key, {}),
                "augmented": True,
                "parent_group": group_key,
                "augmentation_method": "nearest_neighbor",
                "instances": len(new_motifs),
            }

        # Update library
        self.semantic_motifs.update(augmented_motifs)
        self.stats["total_augmented_motifs"] = sum(
            len(motifs) for key, motifs in augmented_motifs.items()
        )

        return augmented_motifs

    def augment_by_bootstrapping(self, num_samples=10):
        """
        Augment the motif library using bootstrapping.

        Generates new synthetic motifs by resampling from the distribution of node/edge features.

        Args:
            num_samples: Number of augmented samples to generate per motif group

        Returns:
            Dictionary of augmented motifs
        """
        if not self.semantic_motifs:
            print("Error: No motifs in the library to augment. Extract motifs first.")
            return {}

        print(
            f"\nPerforming bootstrapping augmentation ({num_samples} samples per group)..."
        )
        augmented_motifs = {}

        # Collect all node labels across motifs
        all_labels = []
        for motifs in self.semantic_motifs.values():
            for motif in motifs:
                all_labels.extend(
                    [motif.nodes[n].get("label", "") for n in motif.nodes()]
                )

        # Remove empty labels and duplicates while preserving order
        seen = set()
        all_labels = [
            label
            for label in all_labels
            if label and label not in seen and not seen.add(label)
        ]

        # Process each semantic group
        for group_key, motifs in tqdm(
            self.semantic_motifs.items(), desc="Bootstrapping groups"
        ):
            if "augmented" in group_key:
                # Skip already augmented groups
                continue

            # Get a representative motif
            base_motif = motifs[0] if motifs else None
            if not base_motif:
                continue

            # Create new synthetic motifs
            new_motifs = []

            for _ in range(num_samples):
                # Create a copy of the structure
                synthetic_motif = nx.DiGraph()
                synthetic_motif.add_nodes_from(base_motif.nodes())
                synthetic_motif.add_edges_from(base_motif.edges())

                # Assign bootstrapped labels
                for node in synthetic_motif.nodes():
                    # Randomly choose a label from the corpus with 80% probability,
                    # or keep the original label with 20% probability
                    if random.random() < 0.8:
                        synthetic_motif.nodes[node]["label"] = random.choice(all_labels)
                    else:
                        synthetic_motif.nodes[node]["label"] = base_motif.nodes[
                            node
                        ].get("label", "")

                new_motifs.append(synthetic_motif)

            # Add the augmented motifs to the library
            aug_key = f"{group_key}_bootstrapped"
            augmented_motifs[aug_key] = new_motifs

            # Add metadata
            self.motif_metadata[aug_key] = {
                **self.motif_metadata.get(group_key, {}),
                "augmented": True,
                "parent_group": group_key,
                "augmentation_method": "bootstrapping",
                "instances": len(new_motifs),
            }

        # Update library
        self.semantic_motifs.update(augmented_motifs)
        num_bootstrapped = sum(len(motifs) for key, motifs in augmented_motifs.items())
        self.stats["total_augmented_motifs"] += num_bootstrapped

        return augmented_motifs

    def get_motif_summary(self):
        """
        Get a summary of all motifs in the library.

        Returns:
            Dictionary with summary information
        """
        summary = {
            "stats": self.stats,
            "motif_types": {
                k: {"description": v, "count": 0} for k, v in self.motif_types.items()
            },
            "semantic_groups": len(self.motif_metadata),
            "augmented_groups": sum(
                1
                for meta in self.motif_metadata.values()
                if meta.get("augmented", False)
            ),
            "total_motifs": sum(
                len(motifs) for motifs in self.semantic_motifs.values()
            ),
        }

        # Count by motif type
        for key, meta in self.motif_metadata.items():
            motif_type = meta.get("motif_type")
            if motif_type in summary["motif_types"]:
                summary["motif_types"][motif_type]["count"] += meta.get("instances", 0)

        return summary

    def visualize_motif_group(self, group_key, output_dir="output", max_examples=3):
        """
        Visualize a group of semantically similar motifs.

        Args:
            group_key: Key for the motif group
            output_dir: Directory to save visualization
            max_examples: Maximum number of example motifs to show
        """
        if group_key not in self.semantic_motifs:
            print(f"Group '{group_key}' not found in the library.")
            return

        motifs = self.semantic_motifs[group_key]
        if not motifs:
            print(f"No motifs in group '{group_key}'.")
            return

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Limit the number of examples
        examples = motifs[: min(max_examples, len(motifs))]

        # Create a figure with subplots for each example
        fig, axes = plt.subplots(1, len(examples), figsize=(6 * len(examples), 5))
        if len(examples) == 1:
            axes = [axes]

        # Get motif type from metadata
        metadata = self.motif_metadata.get(group_key, {})
        motif_type = metadata.get("motif_type", group_key.split("_")[0])
        motif_name = metadata.get(
            "description", self.motif_types.get(motif_type, "Unknown Type")
        )

        # Plot each example
        for i, subgraph in enumerate(examples):
            ax = axes[i]

            # Create node labels
            node_labels = {
                node: subgraph.nodes[node].get("label", str(node))
                for node in subgraph.nodes()
            }

            # Simple coloring scheme based on node position in the graph
            in_degree = dict(subgraph.in_degree())
            out_degree = dict(subgraph.out_degree())

            node_colors = []
            for node in subgraph.nodes():
                # Source nodes (no incoming edges)
                if in_degree[node] == 0 and out_degree[node] > 0:
                    node_colors.append("skyblue")
                # Sink nodes (no outgoing edges)
                elif in_degree[node] > 0 and out_degree[node] == 0:
                    node_colors.append("salmon")
                # Intermediate nodes
                else:
                    node_colors.append("lightgreen")

            # Draw the graph
            pos = nx.spring_layout(subgraph, seed=42)
            nx.draw_networkx_nodes(
                subgraph, pos, ax=ax, node_color=node_colors, node_size=1500, alpha=0.8
            )
            nx.draw_networkx_edges(
                subgraph, pos, ax=ax, width=2, edge_color="gray", arrowsize=20
            )
            nx.draw_networkx_labels(
                subgraph, pos, labels=node_labels, ax=ax, font_size=10
            )

            # Add a title
            if i == 0:
                ax.set_title(
                    f"{motif_name} ({motif_type}): Group {group_key}\nSample {i+1}/{len(motifs)} examples"
                )
            else:
                ax.set_title(f"Sample {i+1}/{len(motifs)} examples")

            ax.axis("off")

        # Save the figure
        plt.tight_layout()
        filename = f"motif_group_{group_key.replace('/', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

        return os.path.join(output_dir, filename)

    def visualize_all_groups(
        self, output_dir="output", max_groups=None, max_examples=2
    ):
        """
        Visualize all motif groups in the library.

        Args:
            output_dir: Directory to save visualizations
            max_groups: Maximum number of groups to visualize (None=all)
            max_examples: Maximum number of example motifs per group

        Returns:
            List of saved visualization files
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Sort groups by size (number of motifs)
        sorted_groups = sorted(
            self.semantic_motifs.items(), key=lambda x: len(x[1]), reverse=True
        )

        if max_groups is not None:
            sorted_groups = sorted_groups[:max_groups]

        # Visualize each group
        visualization_files = []
        for group_key, motifs in tqdm(sorted_groups, desc="Visualizing groups"):
            file_path = self.visualize_motif_group(
                group_key, output_dir=output_dir, max_examples=max_examples
            )
            visualization_files.append(file_path)

        return visualization_files

    def save_library(self, file_path):
        """Save the motif library to a file."""
        try:
            # Create a serializable version of the library
            library_data = {
                "min_motif_size": self.min_motif_size,
                "max_motif_size": self.max_motif_size,
                "min_semantic_similarity": self.min_semantic_similarity,
                "stats": self.stats,
                "motif_metadata": self.motif_metadata,
                "semantic_motifs": {
                    group_key: [
                        {
                            "nodes": list(m.nodes()),
                            "edges": list(m.edges()),
                            "node_labels": {
                                str(n): m.nodes[n].get("label", str(n))
                                for n in m.nodes()
                            },
                            "metadata": {
                                "motif_type": m.graph.get("motif_type", "Unknown"),
                                "motif_description": m.graph.get(
                                    "motif_description", "Unknown Type"
                                ),
                                "demographics": m.graph.get(
                                    "metadata", []
                                ),  # List of {agent_id, demographic} dicts
                                "confidence": {
                                    str(n): m.nodes[n].get("confidence", 0)
                                    for n in m.nodes()
                                },
                                "importance": {
                                    str(n): m.nodes[n].get("importance", 0)
                                    for n in m.nodes()
                                },
                            },
                        }
                        for m in group_data[
                            "instances"
                        ]  # Access instances from the group data
                    ]
                    for group_key, group_data in self.semantic_motifs.items()
                },
            }

            # Save to JSON
            with open(file_path, "w") as f:
                json.dump(library_data, f, indent=2)

            print(f"Motif library saved to {file_path}")
            return True

        except Exception as e:
            print(f"Error saving motif library: {e}")
            import traceback

            traceback.print_exc()  # Print the full stack trace for debugging
            return False

    @classmethod
    def load_library(cls, json_path: str) -> "MotifLibrary":
        """Load motif library from JSON file."""
        library = cls()

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            # Load basic parameters
            library.min_motif_size = data.get("min_motif_size", 3)
            library.max_motif_size = data.get("max_motif_size", 5)
            library.min_semantic_similarity = data.get("min_semantic_similarity", 0.4)

            # Load semantic motifs and convert them to NetworkX graphs
            semantic_motifs_data = data.get("semantic_motifs", {})
            for group_key, motifs in semantic_motifs_data.items():
                library.semantic_motifs[group_key] = []
                for motif_data in motifs:
                    G = nx.DiGraph()

                    # Add nodes with labels
                    for node, label in motif_data.get("node_labels", {}).items():
                        G.add_node(node, label=label)

                    # Add edges
                    for edge in motif_data.get("edges", []):
                        if len(edge) >= 2:
                            G.add_edge(edge[0], edge[1])

                    # Add metadata including demographics
                    G.graph["motif_metadata"] = motif_data.get("metadata", {})

                    library.semantic_motifs[group_key].append(G)

            print(
                f"Loaded {len(library.semantic_motifs)} motif groups with "
                f"{sum(len(motifs) for motifs in library.semantic_motifs.values())} total motifs"
            )

            # Print example of demographic data to verify
            for group_key, motifs in library.semantic_motifs.items():
                if motifs:
                    example_motif = motifs[0]
                    metadata = example_motif.graph.get("motif_metadata", {})
                    if metadata.get("demographics"):
                        print("\nExample motif metadata:")
                        print(f"Group: {group_key}")
                        print(f"Metadata: {metadata}")
                    break

        except Exception as e:
            print(f"Error loading motif library: {str(e)}")
            return library

        return library

    def export_to_json(self, file_path):
        """
        Export the motif library summary to JSON.

        Args:
            file_path: Path to save the JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create summary data
            summary_data = {"stats": self.stats, "motif_groups": []}

            # Add metadata for each group
            for group_key, metadata in self.motif_metadata.items():
                group_data = {
                    "group_key": group_key,
                    **metadata,
                    "instances": len(self.semantic_motifs.get(group_key, [])),
                }
                summary_data["motif_groups"].append(group_data)

            # Save to JSON
            with open(file_path, "w") as f:
                json.dump(summary_data, f, indent=2)

            print(f"Motif library summary exported to {file_path}")
            return True

        except Exception as e:
            print(f"Error exporting motif library summary: {e}")
            return False

    def analyze_demographic_distribution(self):
        """
        Analyze the distribution of demographics across motif groups.

        Returns:
            Dict containing demographic statistics for each motif group
        """
        demographic_stats = {}

        for group_key, motifs in self.semantic_motifs.items():
            group_stats = {
                "total_instances": len(motifs),
                "unique_demographics": set(),
                "demographic_counts": defaultdict(int),
            }

            for motif in motifs:
                demographics = motif.graph.get("demographics", [])
                for demo in demographics:
                    if demo:  # Only count non-empty demographics
                        demo_str = str(demo)  # Convert to string for set storage
                        group_stats["unique_demographics"].add(demo_str)
                        group_stats["demographic_counts"][demo_str] += 1

            # Convert sets to lists for JSON serialization
            group_stats["unique_demographics"] = list(
                group_stats["unique_demographics"]
            )
            group_stats["demographic_counts"] = dict(group_stats["demographic_counts"])

            demographic_stats[group_key] = group_stats

        return demographic_stats

    def identify_motif_type(self, G):
        """
        Identify detailed motif type based on the paper's classification:
        M1: Chain (exactly 3 nodes)
        M2.x: Fork patterns (1-to-many)
        M3.x: Collider patterns (many-to-1)
        """
        nodes = list(G.nodes())
        n = len(nodes)

        if n < 3:
            return "Unknown"

        # Find sources and sinks
        sources = [n for n in nodes if G.in_degree(n) == 0]
        sinks = [n for n in nodes if G.out_degree(n) == 0]

        # Chain (M1) - exactly 3 nodes with linear sequence
        if n == 3 and len(sources) == 1 and len(sinks) == 1:
            source = sources[0]
            sink = sinks[0]
            intermediate = [n for n in nodes if n not in [source, sink]][0]
            if G.has_edge(source, intermediate) and G.has_edge(intermediate, sink):
                return "M1"

        # Fork patterns (M2.x)
        if len(sources) == 1:
            source = sources[0]
            out_degree = G.out_degree(source)
            if out_degree >= 2:
                if out_degree == 2:
                    return "M2.1"  # Basic fork
                elif out_degree == 3:
                    return "M2.2"  # Extended fork
                else:
                    return "M2.3"  # Large fork

        # Collider patterns (M3.x)
        if len(sinks) == 1:
            sink = sinks[0]
            in_degree = G.in_degree(sink)
            if in_degree >= 2:
                if in_degree == 2:
                    return "M3.1"  # Basic collider
                elif in_degree == 3:
                    return "M3.2"  # Extended collider
                else:
                    return "M3.3"  # Large collider

        return "Unknown"

    def merge_motifs(self, motif1, motif2, mapping):
        """
        Merge two motifs into a single motif, combining their metadata and demographics.

        Args:
            motif1: First motif (base motif to merge into)
            motif2: Second motif to merge
            mapping: Node mapping from motif1 to motif2

        Returns:
            Merged motif with combined metadata
        """
        # Create a copy of the first motif as our base
        merged_motif = motif1.copy()

        # Merge node attributes
        for node1, node2 in mapping.items():
            # Merge labels if they exist
            label1 = motif1.nodes[node1].get("label", "")
            label2 = motif2.nodes[node2].get("label", "")
            if label1 and label2:
                # Keep the more specific or longer label
                merged_motif.nodes[node1]["label"] = (
                    label1 if len(label1) > len(label2) else label2
                )

            # Merge confidence scores
            conf1 = motif1.nodes[node1].get("confidence", 0.0)
            conf2 = motif2.nodes[node2].get("confidence", 0.0)
            merged_motif.nodes[node1]["confidence"] = max(conf1, conf2)

            # Merge importance scores
            imp1 = motif1.nodes[node1].get("importance", 0.0)
            imp2 = motif2.nodes[node2].get("importance", 0.0)
            merged_motif.nodes[node1]["importance"] = max(imp1, imp2)

        # Merge edge attributes
        for edge1 in motif1.edges():
            edge2 = (mapping[edge1[0]], mapping[edge1[1]])
            if motif2.has_edge(*edge2):
                # Merge edge attributes if they exist
                edge_data1 = motif1.edges[edge1]
                edge_data2 = motif2.edges[edge2]
                merged_edge_data = {
                    k: max(edge_data1.get(k, 0.0), edge_data2.get(k, 0.0))
                    for k in set(edge_data1) | set(edge_data2)
                }
                merged_motif.edges[edge1].update(merged_edge_data)

        # Merge graph-level attributes
        merged_motif.graph.update(
            {
                "demographics": self.get_group_demographics([motif1, motif2]),
                "motif_type": motif1.graph.get("motif_type"),
                "description": motif1.graph.get("description"),
            }
        )

        return merged_motif


def load_graph_from_json(
    graph_data: Dict, agent_id: str = None, demographic_data: Dict = None
) -> nx.DiGraph:
    """
    Load graph data to NetworkX DiGraph with graceful handling of missing fields

    Args:
        graph_data: Dictionary containing graph data
        agent_id: agent's ID
        demographic_data: agent's demographic data
    """
    G = nx.DiGraph()

    # Get actual graph data
    if isinstance(graph_data, dict) and "graph" in graph_data:
        graph_content = graph_data["graph"]
    else:
        print(f"Warning: Invalid graph data format for agent {agent_id}")
        return G

    # Add nodes with safe attribute access
    for node_id, node_data in graph_content.get("nodes", {}).items():
        if not isinstance(node_data, dict):
            print(f"Warning: Invalid node data for node {node_id}")
            continue

        G.add_node(
            node_id,
            label=node_data.get("label", ""),
            confidence=node_data.get(
                "aggregate_confidence", 0.0
            ),  # Default to 0.0 if missing
            importance=node_data.get("importance", 0.0),  # Default to 0.0 if missing
            is_stance=node_data.get("is_stance", False),
            status=node_data.get("status", "unknown"),
        )

    # Add edges with safe attribute access
    for edge_id, edge_data in graph_content.get("edges", {}).items():
        if not isinstance(edge_data, dict):
            print(f"Warning: Invalid edge data for edge {edge_id}")
            continue

        if "source" in edge_data and "target" in edge_data:
            G.add_edge(
                edge_data["source"],
                edge_data["target"],
                id=edge_id,
                confidence=edge_data.get(
                    "aggregate_confidence", 0.0
                ),  # Default to 0.0 if missing
                direction=edge_data.get("direction", ""),
                modifier=edge_data.get("modifier", 0.0),  # Default to 0.0 if missing
            )

    return G


def find_agent_graph_data(responses_file: str) -> Dict[str, Dict]:
    """
    From new JSON format, read all agent's graph data
    """
    try:
        with open(responses_file, "r") as f:
            data = json.load(f)

        return data
    except Exception as e:
        print(f"Error loading graph data: {e}")
        return {}


def get_demographic_statistics(samples_dir: str) -> dict:
    """
    Extract demographic statistics from new data format
    """
    try:
        with open(samples_dir, "r") as f:
            data = json.load(f)

        demographic_stats = {}
        for agent_id, agent_data in data.items():
            if "demographics" in agent_data:
                demographic_stats[agent_id] = agent_data["demographics"]

        return demographic_stats
    except Exception as e:
        print(f"Error loading demographic data: {e}")
        return {}


def load_demographic_data(responses_file: str) -> Dict[str, Dict]:
    """
    Load full agent data from responses file

    Args:
        responses_file: Path to responses JSON file

    Returns:
        Dict[str, Dict]: Mapping from agent_id to its complete data
    """
    try:
        with open(responses_file, "r") as f:
            data = json.load(f)

        demographic_mapping = {}
        # Process list format data
        for agent_data in data:  # Now iterating over list instead of dict
            agent_id = agent_data.get("id")  # Get id from each entry
            if agent_id:
                demographic_mapping[agent_id] = {
                    "has_demographics": True,
                    "agent_data": agent_data,  # Save full agent data
                }
                # print(f"Found agent data for {agent_id}")  # Add debug info

        print(f"Loaded demographic data for {len(demographic_mapping)} agents")
        return demographic_mapping

    except Exception as e:
        print(f"Error loading demographic data: {e}")
        import traceback

        traceback.print_exc()  # Print full error stack
        return {}


def identify_motif_type(G):
    """Identify the type of motif based on graph structure.
    This is a standalone version of MotifLibrary.identify_motif_type
    """
    num_nodes = G.number_of_nodes()
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    # Basic checks
    if num_nodes < 3:
        return "Unknown", "Too small"

    # Chain (M1): Exactly 3 nodes in a sequence
    if num_nodes == 3:
        # Check for chain pattern
        nodes = list(G.nodes())
        if (
            any(out_degrees[n] == 1 and in_degrees[n] == 0 for n in nodes)
            and any(out_degrees[n] == 1 and in_degrees[n] == 1 for n in nodes)
            and any(out_degrees[n] == 0 and in_degrees[n] == 1 for n in nodes)
        ):
            return "M1", "Chain"

    # Fork patterns (M2.x)
    source_nodes = [n for n, d in out_degrees.items() if d >= 2 and in_degrees[n] == 0]
    if len(source_nodes) == 1:
        source = source_nodes[0]
        num_children = out_degrees[source]
        if num_children == 2:
            return "M2.1", "Basic Fork"
        elif num_children == 3:
            return "M2.2", "Extended Fork"
        elif num_children >= 4:
            return "M2.3", "Large Fork"

    # Collider patterns (M3.x)
    sink_nodes = [n for n, d in in_degrees.items() if d >= 2 and out_degrees[n] == 0]
    if len(sink_nodes) == 1:
        sink = sink_nodes[0]
        num_parents = in_degrees[sink]
        if num_parents == 2:
            return "M3.1", "Basic Collider"
        elif num_parents == 3:
            return "M3.2", "Extended Collider"
        elif num_parents >= 4:
            return "M3.3", "Large Collider"

    return "Unknown", "Unknown Type"


def convert_graph_to_dict(G):
    """Convert a NetworkX graph to a serializable dictionary"""
    # Identify motif type using the standalone function
    motif_type, motif_desc = identify_motif_type(G)

    # Extract and flatten demographics from graph metadata
    demographics = []
    if G.graph.get("metadata"):
        demographics = G.graph["metadata"]
    elif G.graph.get("graph_metadata", {}).get("metadata"):
        demographics = G.graph["graph_metadata"]["metadata"]

    # Process nodes with both underscore labels and readable names
    nodes_dict = {}
    for node in G.nodes():
        label = G.nodes[node].get("label", "")
        # Convert label to underscore format
        underscore_label = "_".join(label.lower().split())
        nodes_dict[node] = {
            "label": underscore_label,  # underscore version
            "name": label,  # original readable version
            "confidence": G.nodes[node].get("confidence", 0),
            "importance": G.nodes[node].get("importance", 0),
            "is_stance": G.nodes[node].get("is_stance", False),
            "status": G.nodes[node].get("status", "unknown"),
        }

    # Process edges with their attributes
    edges_list = []
    for source, target, data in G.edges(data=True):
        edge_data = {
            "source": source,
            "target": target,
            "modifier": data.get("modifier", ""),
            "confidence": data.get("confidence", 0),
            "direction": data.get("direction", ""),
            "strength": data.get("strength", 0),
        }
        edges_list.append(edge_data)

    return {
        "nodes": nodes_dict,
        "edges": edges_list,  # Now using the detailed edge data
        "metadata": {
            "demographics": demographics,
            "motif_type": motif_type,
            "motif_description": motif_desc,
        },
    }


def convert_dict_to_graph(data):
    """Convert a dictionary back to a NetworkX graph"""
    G = nx.DiGraph()

    # Add nodes with attributes
    for node, attrs in data["nodes"].items():
        node_attrs = {
            "label": attrs.get("name", ""),  # Use the readable name for the graph
            "underscore_label": attrs.get("label", ""),  # Store underscore version
            "confidence": attrs.get("confidence", 0),
            "importance": attrs.get("importance", 0),
            "is_stance": attrs.get("is_stance", False),
            "status": attrs.get("status", "unknown"),
        }
        G.add_node(node, **node_attrs)

    # Add edges with their attributes
    for edge in data["edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            modifier=edge.get("modifier", ""),
            confidence=edge.get("confidence", 0),
            direction=edge.get("direction", ""),
            strength=edge.get("strength", 0),
        )

    # Add metadata directly to graph
    if "metadata" in data:
        G.graph["metadata"] = data["metadata"].get(
            "demographics", []
        )  # Store demographics directly
        G.graph["motif_type"] = data["metadata"].get("motif_type", "Unknown")
        G.graph["motif_description"] = data["metadata"].get(
            "motif_description", "Unknown Type"
        )

    return G


def process_causal_graphs(
    clean_graphs_file: str,
    responses_file: str,
    output_dir: str = None,
    min_semantic_similarity: float = 0.4,
):
    """Process all causal graphs to extract and analyze motifs"""
    all_topological_motifs = {}

    if output_dir is None:
        output_dir = os.path.dirname(clean_graphs_file)
    os.makedirs(output_dir, exist_ok=True)

    # Load responses data to get demographic information
    demographic_mapping = load_demographic_data(responses_file)

    # Create motif library
    library = MotifLibrary(min_semantic_similarity=min_semantic_similarity)

    # Load and process all graphs
    try:
        with open(clean_graphs_file, "r") as f:
            all_data = json.load(f)
            print(f"Loaded graph data with keys: {list(all_data.keys())[:5]}")
            print(
                f"Example data structure: {list(all_data.values())[0].keys() if all_data else 'Empty'}"
            )

        # # HACK: test on one person only
        # all_data = {
        #     "660cd7b61a24eeff3eac6e93": all_data["660cd7b61a24eeff3eac6e93"],
        #     "664662d0e586193a0ca4267e": all_data["664662d0e586193a0ca4267e"],
        # }

        current_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(current_dir, "sample_graph_motifs.json")):
            with open(os.path.join(current_dir, "sample_graph_motifs.json"), "r") as f:
                all_topological_motifs = json.load(f)

        else:
            # Process each agent's graph
            for agent_id, data in tqdm(all_data.items(), desc="Processing graphs"):
                print(f"\nProcessing agent {agent_id}")

                try:
                    if not isinstance(data, dict):
                        print(f"Skipping agent {agent_id}: Data is not a dictionary")
                        continue

                    if not data.get("graph"):
                        print(f"Skipping agent {agent_id}: No graph data found")
                        continue

                    if not data.get("graph", {}).get("nodes"):
                        print(f"Skipping agent {agent_id}: No nodes in graph")
                        continue

                    # Get demographic info
                    agent_info = demographic_mapping.get(
                        agent_id, {"has_demographics": False}
                    )
                    has_demographics = agent_info.get("has_demographics", False)

                    if has_demographics:
                        print(f"Found demographics for {agent_id}")

                    # Convert to NetworkX graph
                    G = load_graph_from_json(
                        data,
                        agent_id=agent_id,
                        demographic_data=(
                            agent_info.get("agent_data") if has_demographics else None
                        ),
                    )

                    # Extract topological motifs
                    motifs = library.extract_topological_motifs(
                        G,
                        sample_id=agent_id,
                        demographic_info=(
                            agent_info.get("agent_data") if has_demographics else None
                        ),
                    )

                    # Convert motifs to serializable format
                    all_topological_motifs[agent_id] = [
                        convert_graph_to_dict(m) for m in motifs
                    ]
                    print(f"Found {len(motifs)} motifs for agent {agent_id}")

                    # Save intermediate results
                    try:
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        with open(
                            os.path.join(current_dir, "sample_graph_motifs.json"), "w"
                        ) as f:
                            json.dump(all_topological_motifs, f, indent=2)
                    except Exception as e:
                        print(f"Warning: Could not save intermediate results: {e}")

                except Exception as e:
                    print(f"Error processing agent {agent_id}: {e}")
                    continue

        # Convert stored motifs back to graphs before semantic filtering
        converted_motifs = {}
        for agent_id, motifs in all_topological_motifs.items():
            converted_motifs[agent_id] = [convert_dict_to_graph(m) for m in motifs]

        # Process all motifs
        library.apply_semantic_filtering(converted_motifs)

        # Save final results
        library.save_library(os.path.join(output_dir, "motif_library.json"))
        return library

    except Exception as e:
        print(f"Error processing graphs: {e}")
        import traceback

        traceback.print_exc()
        return None


def analyze_demographics(library):
    """Analyze demographic distribution of motifs"""
    demographic_stats = defaultdict(lambda: defaultdict(int))

    for group_key, motifs in library.semantic_motifs.items():
        for motif in motifs:
            demo_info = motif.graph.get("demographics", {})
            for category, value in demo_info.items():
                if value is not None:
                    demographic_stats[category][value] += 1

    return dict(demographic_stats)


def main():
    """
    Main function to run the motif library
    """
    import argparse

    parser = argparse.ArgumentParser(description="Motif Library Builder")
    parser.add_argument(
        "--clean-graphs", required=True, help="Path to clean graphs JSON file"
    )
    parser.add_argument(
        "--responses",
        required=True,
        help="Path to responses file with demographic data",
    )
    parser.add_argument("--output", "-o", help="Output directory for results")
    parser.add_argument(
        "--similarity",
        "-s",
        type=float,
        default=0.4,
        help="Minimum semantic similarity (0-1)",
    )

    args = parser.parse_args()

    # Process all causal graphs
    library = process_causal_graphs(
        args.clean_graphs,
        args.responses,
        args.output,
        min_semantic_similarity=args.similarity,
    )


if __name__ == "__main__":
    main()
