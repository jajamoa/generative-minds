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


from .semantic_similarity import SemanticSimilarityEngine


def ensure_evaluation_prefix(path: str) -> str:
    """Ensure path has src/evaluation prefix if not already present."""
    prefix = "src/evaluation_T3"
    if not path.startswith(prefix) and not path.startswith("/"):
        return os.path.join(prefix, path)
    return path


def convert_raw_data_to_clean_version(raw_data_path: str, clean_data_path: str, agent_demographic_path: str):
    with open(raw_data_path, "r") as f:
        raw_data = json.load(f)
    
    with open(agent_demographic_path, "r") as f:
        agent_demographic = json.load(f)

    agent_ids =  [agent["id"] for agent in agent_demographic]

    clean_data = {}
    for user_data in raw_data:
        if user_data["prolificId"] not in agent_ids:
            continue
        clean_data[user_data["prolificId"]] = {
            "graph": user_data["graphs"][0]["graphData"],
            "stance_nodes": [user_data["graphs"][0]["graphData"]["stance_node_id"]],
        }

        clean_data[user_data["prolificId"]].pop("agent_id")

    with open(clean_data_path, "w") as f:
        json.dump(clean_data, f)


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

        # Validate node mapping
        try:
            # Check if all mapped nodes exist in both graphs
            for node1, node2 in node_mapping.items():
                if node1 not in motif1.nodes or node2 not in motif2.nodes:
                    print(f"Warning: Invalid node mapping {node1}->{node2}")
                    return 0.0
        except Exception as e:
            print(f"Error validating node mapping: {e}")
            return 0.0

        # Position weights as specified in the paper
        weights = {"source": 1.5, "sink": 1.5, "intermediate": 1.0}

        # Get node positions
        positions1 = self.compute_node_position_info(motif1)
        positions2 = self.compute_node_position_info(motif2)

        total_weighted_sim = 0.0
        total_weight = 0.0

        for node1, node2 in node_mapping.items():
            try:
                # Get node names with safe fallback
                name1 = motif1.nodes[node1].get(
                    "name", motif1.nodes[node1].get("label", str(node1))
                )
                name2 = motif2.nodes[node2].get(
                    "name", motif2.nodes[node2].get("label", str(node2))
                )

                # Print node names being compared for debugging
                print(f"\nMerging motifs:")
                print(
                    f"Motif 1 nodes: {[motif1.nodes[n].get('name', '') for n in motif1.nodes]}"
                )
                print(
                    f"Motif 2 nodes: {[motif2.nodes[n].get('name', '') for n in motif2.nodes]}"
                )
                print(f"Node mapping: {node_mapping}")

                # Calculate semantic similarity using names
                sim = self.similarity_engine.node_similarity(name1, name2)

                # Get position weights with safe fallback
                pos1 = positions1.get(node1, "intermediate")
                pos2 = positions2.get(node2, "intermediate")
                weight = (weights[pos1] + weights[pos2]) / 2

                total_weighted_sim += weight * sim
                total_weight += weight

            except Exception as e:
                print(f"Error processing node pair {node1}->{node2}: {e}")
                continue

        return total_weighted_sim / total_weight if total_weight > 0 else 0.0

    def apply_semantic_filtering(self, topological_motifs=None):
        """
        Apply semantic filtering across all agents' motifs at once.
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
        }

        # Step 1: Collect ALL motifs from all agents into a single pool
        print("\nPhase 1: Collecting and cleaning all motifs...")
        all_motifs = []
        for agent_id, agent_motifs in topological_motifs.items():
            for motif in agent_motifs:
                cleaned_motif = self.clean_motif(motif)
                # Store agent ID in the motif metadata
                cleaned_motif.graph["agent_id"] = agent_id
                all_motifs.append(cleaned_motif)
                stats["total_input_motifs"] += 1

        # Step 2: Group by motif type
        print("\nPhase 2: Grouping by motif type...")
        typed_motifs = defaultdict(list)
        for motif in all_motifs:
            motif_type = self.identify_motif_type(motif)
            typed_motifs[motif_type].append(motif)
            stats["motifs_per_type"][motif_type] += 1

        # Step 3: Process each type group
        print("\nPhase 3: Semantic clustering within type groups...")
        for motif_type, motifs in typed_motifs.items():
            print(f"\nProcessing {motif_type} motifs ({len(motifs)} instances)...")

            # Skip if only one motif
            if len(motifs) <= 1:
                group_key = f"{motif_type}_1"
                abstract_motif = self.create_abstract_motif(
                    motifs
                )  # Create abstract motif even for single motif
                semantic_groups[group_key] = {
                    "instances": motifs,
                    "abstract_motif": abstract_motif,
                    "demographics": self.get_group_demographics(motifs),
                }
                continue

            # Create clusters of similar motifs
            clusters = self.cluster_similar_motifs(motifs)

            # Create semantic groups from clusters
            for cluster_id, cluster_motifs in enumerate(clusters):
                group_key = f"{motif_type}_{cluster_id + 1}"
                abstract_motif = self.create_abstract_motif(cluster_motifs)
                semantic_groups[group_key] = {
                    "instances": cluster_motifs,
                    "abstract_motif": abstract_motif,
                    "demographics": self.get_group_demographics(cluster_motifs),
                }
                stats["groups_per_type"][motif_type] += 1
                stats["total_groups"] += 1

        # Update library
        self.semantic_motifs = semantic_groups
        self.stats.update(stats)

        return semantic_groups

    def cluster_similar_motifs(self, motifs):
        """
        Cluster similar motifs using structural and semantic similarity.
        Uses a single-pass clustering approach for efficiency.
        """
        clusters = []
        processed = set()

        for i, motif1 in enumerate(motifs):
            if i in processed:
                continue

            # Start new cluster
            current_cluster = [motif1]
            processed.add(i)

            # Find similar motifs
            for j, motif2 in enumerate(motifs):
                if j in processed:
                    continue

                try:
                    # Check isomorphism first (fast structural check)
                    matcher = nx.algorithms.isomorphism.DiGraphMatcher(motif1, motif2)
                    if matcher.is_isomorphic():
                        mapping = next(matcher.isomorphisms_iter())

                    # Verify structural and semantic equivalence
                    if self.verify_structural_equivalence(
                        motif1, motif2, mapping
                    ) and self.verify_semantic_equivalence(motif1, motif2, mapping):
                        current_cluster.append(motif2)
                        processed.add(j)

                except Exception as e:
                    print(f"Error comparing motifs {i} and {j}: {e}")
                    continue

            clusters.append(current_cluster)

        return clusters

    def create_abstract_motif(self, cluster_motifs):
        """
        Create an abstract representation of a cluster of similar motifs.
        """
        if not cluster_motifs:
            return None

        # Use first motif as base
        base_motif = cluster_motifs[0]
        abstract_motif = base_motif.copy()

        # Collect all alternative labels and metadata
        for node in abstract_motif.nodes():
            abstract_motif.nodes[node]["alternative_labels"] = set()
            abstract_motif.nodes[node]["confidence"] = []
            abstract_motif.nodes[node]["importance"] = []

        # Initialize combined metadata
        combined_metadata = []
        combined_demographics = []

        # Merge information from all motifs in cluster
        for motif in cluster_motifs:
            # Collect metadata
            if "metadata" in motif.graph:
                combined_metadata.append(motif.graph["metadata"])
            if "demographics" in motif.graph:
                combined_demographics.extend(motif.graph["demographics"])

            # For the first motif, just copy its metadata structure
            if motif == base_motif:
                continue

            # Find mapping to base motif
            matcher = nx.algorithms.isomorphism.DiGraphMatcher(base_motif, motif)
            mapping = next(matcher.isomorphisms_iter())

            # Update node attributes
            for node1, node2 in mapping.items():
                node_data = abstract_motif.nodes[node1]
                other_data = motif.nodes[node2]

                # Collect alternative labels
                node_data["alternative_labels"].add(other_data.get("label", ""))

                # Collect metrics
                node_data["confidence"].append(other_data.get("confidence", 0))
                node_data["importance"].append(other_data.get("importance", 0))

        # Finalize metrics
        for node in abstract_motif.nodes():
            node_data = abstract_motif.nodes[node]
            node_data["alternative_labels"] = list(node_data["alternative_labels"])
            node_data["confidence"] = (
                max(node_data["confidence"]) if node_data["confidence"] else 0
            )
            node_data["importance"] = (
                max(node_data["importance"]) if node_data["importance"] else 0
            )

        # Add combined metadata to abstract motif
        abstract_motif.graph["metadata"] = combined_metadata
        abstract_motif.graph["demographics"] = combined_demographics
        abstract_motif.graph["motif_type"] = base_motif.graph.get(
            "motif_type", "Unknown"
        )
        abstract_motif.graph["motif_description"] = base_motif.graph.get(
            "motif_description", "Unknown Type"
        )

        return abstract_motif

    def clean_motif(self, motif):
        """
        Clean and standardize a motif before processing
        """
        cleaned = motif.copy()

        # Standardize node labels
        for node in cleaned.nodes():
            label = cleaned.nodes[node].get("label", "")
            if label:
                # Convert to underscore format
                underscore_label = "_".join(label.lower().split())
                cleaned.nodes[node]["label"] = underscore_label
                cleaned.nodes[node]["name"] = label  # Keep original label

            # Ensure all required attributes exist
            for attr in ["confidence", "importance", "is_stance"]:
                if attr not in cleaned.nodes[node]:
                    cleaned.nodes[node][attr] = 0.0 if attr != "is_stance" else False

        # Standardize edge attributes
        for u, v in cleaned.edges():
            for attr in ["modifier", "confidence", "strength"]:
                if attr not in cleaned.edges[u, v]:
                    cleaned.edges[u, v][attr] = 0.0

        return cleaned

    def merge_similar_nodes(self, motif):
        """
        Merge nodes with similar labels within a motif
        """
        merged_motif = motif.copy()
        nodes = list(merged_motif.nodes())
        merged = set()

        for i, node1 in enumerate(nodes):
            if node1 in merged:
                continue

            label1 = merged_motif.nodes[node1].get("label", "")
            for j, node2 in enumerate(nodes[i + 1 :], i + 1):
                if node2 in merged:
                    continue

                label2 = merged_motif.nodes[node2].get("label", "")
                if label1 and label2:
                    sim = self.similarity_engine.node_similarity(label1, label2)
                    if sim >= self.min_semantic_similarity:
                        # Merge node2 into node1
                        # Combine edges
                        for pred in merged_motif.predecessors(node2):
                            if pred != node1:
                                merged_motif.add_edge(pred, node1)
                        for succ in merged_motif.successors(node2):
                            if succ != node1:
                                merged_motif.add_edge(node1, succ)
                        # Remove node2
                        merged_motif.remove_node(node2)
                        merged.add(node2)

        return merged_motif

    def compute_edge_similarity(self, motif1, motif2, node_mapping):
        """
        Compute similarity between edge attributes of two motifs, considering modifiers
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

                        # Calculate modifier similarity (considering both value and sign)
                        mod1 = edge1.get("modifier", 0)
                        mod2 = edge2.get("modifier", 0)
                        # Check if modifiers have same sign (both positive or both negative)
                        same_direction = mod1 * mod2 >= 0
                        modifier_sim = 1.0 if same_direction else 0.0

                        # Calculate confidence similarity
                        conf_sim = 1.0 - abs(
                            edge1.get("confidence", 0) - edge2.get("confidence", 0)
                        )

                        # Weighted combination with higher weight on modifier similarity
                        edge_sim = 0.7 * modifier_sim + 0.3 * conf_sim
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
        merged_demos = {}

        for motif in motifs:
            # Try both 'metadata' and direct demographics access
            demos = motif.graph.get("metadata", [])
            if not demos:
                demos = motif.graph.get("demographics", [])

            if isinstance(demos, list):
                for demo in demos:
                    if isinstance(demo, dict) and "agent_id" in demo:
                        agent_id = demo["agent_id"]
                        if agent_id not in merged_demos:
                            merged_demos[agent_id] = demo
                            print(f"Added demographics for agent {agent_id}")

        demographics = list(merged_demos.values())
        print(f"Total unique demographics found: {len(demographics)}")
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

        print(f"\nDebug: Loading motif library from {json_path}")
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                print(f"Successfully loaded JSON data")
                print(f"Keys in JSON: {list(data.keys())}")
                print(
                    f"Number of semantic motifs in JSON: {len(data.get('semantic_motifs', {}))}"
                )

            # Load basic parameters
            library.min_motif_size = data.get("min_motif_size", 3)
            library.max_motif_size = data.get("max_motif_size", 5)
            library.min_semantic_similarity = data.get("min_semantic_similarity", 0.4)
            library.stats = data.get("stats", {})
            library.motif_metadata = data.get("motif_metadata", {})

            # Load semantic motifs and convert them to NetworkX graphs
            semantic_motifs_data = data.get("semantic_motifs", {})
            print(f"\nProcessing semantic motifs:")
            print(f"Number of motif groups found: {len(semantic_motifs_data)}")

            for group_key, motifs in semantic_motifs_data.items():
                library.semantic_motifs[group_key] = []

                for motif_data in motifs:
                    try:
                        G = nx.DiGraph()

                        # Add nodes with all attributes
                        for node in motif_data.get("nodes", []):
                            node_attrs = {
                                "label": motif_data["node_labels"].get(node, ""),
                                "confidence": motif_data["metadata"]["confidence"].get(
                                    node, 0.0
                                ),
                                "importance": motif_data["metadata"]["importance"].get(
                                    node, 0.0
                                ),
                            }
                            G.add_node(node, **node_attrs)

                        # Add edges
                        for edge in motif_data.get("edges", []):
                            if len(edge) >= 2:
                                G.add_edge(edge[0], edge[1])

                            # Add metadata to graph attributes
                            G.graph["metadata"] = motif_data.get("metadata", {})
                            G.graph["motif_type"] = motif_data["metadata"].get(
                                "motif_type", "Unknown"
                            )
                            G.graph["motif_description"] = motif_data["metadata"].get(
                                "motif_description", "Unknown"
                            )
                            G.graph["demographics"] = motif_data["metadata"].get(
                                "demographics", []
                            )

                        library.semantic_motifs[group_key].append(G)

                    except Exception as e:
                        print(f"Error processing motif in group {group_key}: {str(e)}")
                        continue

            print(f"\nFinal library state:")
            print(f"Number of motif groups: {len(library.semantic_motifs)}")
            print(
                f"Total motifs: {sum(len(motifs) for motifs in library.semantic_motifs.values())}"
            )

        except Exception as e:
            print(f"Error loading motif library: {str(e)}")
            import traceback

            traceback.print_exc()
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
        Merge two motifs only if they represent the same semantic pattern.

        Args:
            motif1: First motif (NetworkX DiGraph)
            motif2: Second motif (NetworkX DiGraph)
            mapping: Node mapping from motif1 to motif2

        Returns:
            Merged motif if the motifs are semantically equivalent, None otherwise
        """
        # First verify structural equivalence
        if not self.verify_structural_equivalence(motif1, motif2, mapping):
            return None

        # Then verify semantic equivalence of mapped nodes
        if not self.verify_semantic_equivalence(motif1, motif2, mapping):
            return None

        # If both checks pass, create merged motif
        merged_motif = nx.DiGraph()

        # Create new node IDs and copy attributes
        new_node_mapping = {}
        for i, (node1, node2) in enumerate(mapping.items()):
            new_id = f"n{i}"
            new_node_mapping[node1] = new_id

            # Merge node attributes
            attrs1 = dict(motif1.nodes[node1])
            attrs2 = dict(motif2.nodes[node2])

            merged_attrs = {
                "label": attrs1.get("label"),  # Keep label from first motif
                "alternative_labels": [attrs2.get("label")],  # Store alternative label
                "confidence": max(
                    attrs1.get("confidence", 0), attrs2.get("confidence", 0)
                ),
                "importance": max(
                    attrs1.get("importance", 0), attrs2.get("importance", 0)
                ),
                "is_stance": attrs1.get("is_stance", False)
                or attrs2.get("is_stance", False),
            }

            merged_motif.add_node(new_id, **merged_attrs)

        # Copy edges and their attributes
        for edge in motif1.edges():
            source, target = edge
            new_source = new_node_mapping[source]
            new_target = new_node_mapping[target]

            # Get edge attributes from both motifs
            edge_data1 = motif1.edges[edge]
            mapped_edge = (mapping[source], mapping[target])
            edge_data2 = motif2.edges[mapped_edge]

            # Merge edge attributes
            merged_edge_attrs = {
                "confidence": max(
                    edge_data1.get("confidence", 0), edge_data2.get("confidence", 0)
                ),
                # Store both modifiers as they might represent different perspectives
                "modifiers": [
                    edge_data1.get("modifier", 0),
                    edge_data2.get("modifier", 0),
                ],
            }

            merged_motif.add_edge(new_source, new_target, **merged_edge_attrs)

        # Merge metadata and demographics
        merged_motif.graph["metadata"] = motif1.graph.get(
            "metadata", []
        ) + motif2.graph.get("metadata", [])

        return merged_motif

    def verify_structural_equivalence(self, motif1, motif2, mapping):
        """
        Verify that two motifs are structurally equivalent under the given mapping.

        Args:
            motif1: First motif
            motif2: Second motif
            mapping: Node mapping from motif1 to motif2

        Returns:
            bool: True if motifs are structurally equivalent
        """
        # Check node count
        if len(motif1) != len(motif2) or len(mapping) != len(motif1):
            return False

        # Check edge preservation
        for edge in motif1.edges():
            source, target = edge
            if not motif2.has_edge(mapping[source], mapping[target]):
                return False

        # Check that no additional edges exist in motif2
        for edge in motif2.edges():
            source, target = edge
            reverse_mapping = {v: k for k, v in mapping.items()}
            if not motif1.has_edge(reverse_mapping[source], reverse_mapping[target]):
                return False

        return True

    def verify_semantic_equivalence(self, motif1, motif2, mapping):
        """
        Verify that mapped nodes are semantically equivalent.

        Args:
            motif1: First motif
            motif2: Second motif
            mapping: Node mapping from motif1 to motif2

        Returns:
            bool: True if nodes are semantically equivalent
        """
        for node1, node2 in mapping.items():
            label1 = motif1.nodes[node1].get("label", "")
            label2 = motif2.nodes[node2].get("label", "")

            # Check semantic similarity using the similarity engine
            similarity = self.similarity_engine.node_similarity(label1, label2)
            if similarity < self.min_semantic_similarity:
                return False

        return True


def load_graph_from_json(
    graph_data: Dict, agent_id: str = None, demographic_data: Dict = None
) -> nx.DiGraph:
    """
    Load graph data to NetworkX DiGraph with graceful handling of missing fields
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
            confidence=node_data.get("aggregate_confidence", 0.0),
            importance=node_data.get("importance", 0.0),
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
                confidence=edge_data.get("aggregate_confidence", 0.0),
                direction=edge_data.get("direction", ""),
                modifier=edge_data.get("modifier", 0.0),
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
    raw_graphs_file: str,
    clean_graphs_file: str,
    agent_demographic_path: str,
    responses_file: str,
    output_path: str = None,
    min_semantic_similarity: float = 0.4,
):
    """Process all causal graphs to extract and analyze motifs"""
    all_topological_motifs = {}


    # Load responses data to get demographic information
    demographic_mapping = load_demographic_data(responses_file)

    # Create motif library
    library = MotifLibrary(min_semantic_similarity=min_semantic_similarity)

    # Load and process all graphs
    try:
        clean_graphs = convert_raw_data_to_clean_version(raw_graphs_file, clean_graphs_file, agent_demographic_path)

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
        #     "5ae5b7e26735bd0001d7dfe3": all_data["5ae5b7e26735bd0001d7dfe3"],
        #     "67e6b4c78b8b457f3bc6a866": all_data["67e6b4c78b8b457f3bc6a866"],
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
        library.save_library(output_path)
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
        "--raw-graphs",
        required=True,
        help="Path to raw graphs JSON file",
    )
    parser.add_argument(
        "--clean-graphs", required=True, help="Path to clean graphs JSON file"
    )
    parser.add_argument(
        "--agent-demographic", required=True, help="Path to agent demographic JSON file"
    )
    parser.add_argument(
        "--responses",
        required=True,
        help="Path to responses file with demographic data",
    )
    parser.add_argument("--output_path", "-o", help="Output path for results")
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
        args.raw_graphs,
        args.clean_graphs,
        args.agent_demographic,
        args.responses,
        args.output_path,
        min_semantic_similarity=args.similarity,
    )


if __name__ == "__main__":
    main()
