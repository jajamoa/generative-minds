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
from semantic_similarity import SemanticSimilarityEngine

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
        self.semantic_motifs = {}     # Dictionary keyed by semantic group ID
        self.motif_metadata = {}      # Store metadata for each motif group
        self.motif_demographics = {}  # Store demographic info for each motif
        self.demographic_distribution = {}  # Overall demographic distribution
        
        # Initialize semantic similarity engine
        self.similarity_engine = SemanticSimilarityEngine(use_wordnet=True)
        
        # Define basic motif templates by type
        self.motif_types = {
            "M1": "Chain",          # A → B → C
            "M2.1": "Basic Fork",   # A → B, A → C (1-to-2)
            "M2.2": "Extended Fork", # A → B, A → C, A → D (1-to-3)
            "M2.3": "Large Fork",   # A → B, A → C, A → D, A → E, ... (1-to-4+)
            "M3.1": "Basic Collider", # A → C, B → C (2-to-1)
            "M3.2": "Extended Collider", # A → D, B → D, C → D (3-to-1)
            "M3.3": "Large Collider"  # A → E, B → E, C → E, D → E, ... (4+-to-1)
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
            for i in range(size-1):
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
            for i in range(size-1):
                G.add_edge(f"n{i}", sink_node)
        
        # Add dummy labels to nodes
        for node in G.nodes():
            G.nodes[node]['label'] = f"node_{node}"
        
        return G
    

    def get_demographic_statistics(samples_dir: str) -> dict:
        demographic_stats = {}
        total_samples = 0
        
        for filename in os.listdir(samples_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(samples_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    metadata = data.get("metadata", {})
                    demographic = metadata.get("perspective", "unknown")
                    
                    if demographic not in demographic_stats:
                        demographic_stats[demographic] = {
                            "count": 0,
                            "samples": []
                        }
                    
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
            "unique_demographics": list(demographic_stats.keys())
        }
    
    def extract_topological_motifs(self, G, motif_types=None, sample_id=None):
        """
        Extract motifs from a graph using topology-based analysis with demographic tracking.
        
        Args:
            G: NetworkX DiGraph to analyze
            motif_types: List of motif types to search for (default: all)
            sample_id: Identifier for the sample (for demographic tracking)
            
        Returns:
            Dictionary of found motifs grouped by type and size
        """
        if motif_types is None:
            motif_types = list(self.motif_types.keys())
        
        topological_motifs = defaultdict(list)
        
        demographic = G.graph.get("metadata", {}).get("perspective", "unknown")
        
        if demographic not in self.demographic_distribution:
            self.demographic_distribution[demographic] = 0
        self.demographic_distribution[demographic] += 1
        
        # Step 1: Track central nodes of each motif type
        # We only track center nodes, not all nodes in motifs
        fork_centers = set()     # Track centers of fork patterns (M2.x)
        collider_centers = set() # Track centers of collider patterns (M3.x)
        
        # Process motif types in order of complexity (larger motifs first)
        ordered_motif_types = [
            # Larger fork patterns first
            "M2.3", "M2.2", "M2.1",
            # Larger collider patterns first
            "M3.3", "M3.2", "M3.1",
            # Chain patterns last
            "M1"
        ]
        
        # Filter out motif types not requested
        ordered_motif_types = [mt for mt in ordered_motif_types if mt in motif_types]
        
        # For each motif type and size (in descending order of size)
        for motif_type in ordered_motif_types:
            print(f"Searching for {motif_type} motifs...")
            
            # Process sizes from large to small
            # For M1 (chain), only process size 3
            if motif_type == "M1":
                sizes = [3]  # Only size 3 for chains
            else:
                sizes = range(self.max_motif_size, self.min_motif_size - 1, -1)
                
            for size in sizes:
                # Create template for this motif type and size
                template = self.create_motif_template(motif_type, size)
                
                # Use VF2 algorithm to find all subgraph matches
                matcher = nx.algorithms.isomorphism.DiGraphMatcher(G, template)
                
                # Find all subgraph isomorphisms (limited to prevent excessive matches)
                max_matches = 30  # Limit to prevent too many matches
                match_count = 0
                
                # Collect subgraph instances
                subgraphs = []
                
                for mapping in matcher.subgraph_isomorphisms_iter():
                    # Mapping is from G to template, we need the reverse
                    # Create a reverse mapping from template to G
                    reverse_mapping = {v: k for k, v in mapping.items()}
                    
                    # Extract the subgraph nodes
                    nodes = [reverse_mapping[n] for n in template.nodes()]
                    
                    # Handle different motif types differently
                    if motif_type.startswith("M2"):  # Fork patterns
                        # For fork patterns, the first node is the central node
                        central_node = nodes[0]
                        
                        # Skip if this center node is already part of a larger fork
                        if central_node in fork_centers:
                            continue
                        
                        # Extract the subgraph
                        subgraph = G.subgraph(nodes).copy()
                        
                        # Add demographic information to the subgraph
                        subgraph.graph["demographic"] = demographic
                        subgraph.graph["sample_id"] = sample_id
                        
                        subgraphs.append(subgraph)
                        
                        # Mark central node as processed for future fork patterns
                        fork_centers.add(central_node)
                    
                    elif motif_type.startswith("M3"):  # Collider patterns
                        # For collider patterns, find the node with highest in-degree
                        in_degrees = {n: G.in_degree(n) for n in nodes}
                        central_node = max(in_degrees, key=in_degrees.get)
                        
                        # Skip if this center node is already part of a larger collider
                        if central_node in collider_centers:
                            continue
                        
                        # Extract the subgraph
                        subgraph = G.subgraph(nodes).copy()
                        
                        # Add demographic information to the subgraph
                        subgraph.graph["demographic"] = demographic
                        subgraph.graph["sample_id"] = sample_id
                        
                        subgraphs.append(subgraph)
                        
                        # Mark central node as processed for future collider patterns
                        collider_centers.add(central_node)
                    
                    else:  # Chain patterns (M1) and others
                        # For chains, we only accept size 3
                        if len(nodes) == 3:
                            subgraph = G.subgraph(nodes).copy()
                            
                            # Add demographic information to the subgraph
                            subgraph.graph["demographic"] = demographic
                            subgraph.graph["sample_id"] = sample_id
                            
                            subgraphs.append(subgraph)
                    
                    match_count += 1
                    if match_count >= max_matches:
                        print(f"  Reached limit of {max_matches} matches for {motif_type}_size_{size}")
                        break
                
                # Add the found subgraphs to the result dictionary
                if subgraphs:
                    key = f"{motif_type}_size_{size}"
                    topological_motifs[key] = subgraphs
                    print(f"  Found {len(subgraphs)} instances of {key}")
                    
                    # 记录每个motif的demographic信息
                    full_key = f"{sample_id}_{key}" if sample_id else key
                    if full_key not in self.motif_demographics:
                        self.motif_demographics[full_key] = []
                    
                    # 为每个subgraph记录demographic
                    for subgraph in subgraphs:
                        self.motif_demographics[full_key].append({
                            "demographic": demographic,
                            "sample_id": sample_id,
                            "motif_type": motif_type,
                            "size": size
                        })
        
        # Remove empty entries
        topological_motifs = {k: v for k, v in topological_motifs.items() if v}
        
        # Update library
        self.topological_motifs.update(topological_motifs)
        self.stats["total_topological_motifs"] = sum(len(motifs) for motifs in self.topological_motifs.values())
        
        return topological_motifs
    
    def compute_node_position_info(self, motif):
        """
        Compute position information for nodes in a motif.
        
        Args:
            motif: NetworkX DiGraph representing a motif
            
        Returns:
            Dictionary mapping node IDs to position information
        """
        in_degrees = dict(motif.in_degree())
        out_degrees = dict(motif.out_degree())
        
        positions = {}
        for node in motif.nodes():
            if in_degrees[node] == 0 and out_degrees[node] > 0:
                positions[node] = "source"
            elif in_degrees[node] > 0 and out_degrees[node] == 0:
                positions[node] = "sink"
            elif in_degrees[node] > 1:
                positions[node] = "collector"
            elif out_degrees[node] > 1:
                positions[node] = "distributor"
            else:
                positions[node] = "intermediate"
        
        return positions
    
    def compute_enhanced_similarity(self, motif1, motif2, node_mapping):
        """
        Compute enhanced semantic similarity between motifs with position weighting.
        
        Args:
            motif1: First motif
            motif2: Second motif
            node_mapping: Node mapping from motif1 to motif2
            
        Returns:
            Similarity score (0-1)
        """
        # Check that the mapping is valid
        if not node_mapping or len(node_mapping) == 0:
            return 0.0
        
        # Get node positions
        positions1 = self.compute_node_position_info(motif1)
        positions2 = self.compute_node_position_info(motif2)
        
        # Calculate node-by-node similarity with position weighting
        similarities = []
        position_weights = []
        
        for node1, node2 in node_mapping.items():
            label1 = motif1.nodes[node1].get('label', str(node1))
            label2 = motif2.nodes[node2].get('label', str(node2))
            
            # Calculate basic similarity
            sim = self.similarity_engine.node_similarity(label1, label2)
            
            # Add position similarity bonus (0.1) if positions match
            pos1 = positions1.get(node1, "unknown")
            pos2 = positions2.get(node2, "unknown")
            if pos1 == pos2:
                sim = min(1.0, sim + 0.1)
            
            # Weight by position importance
            if pos1 in ["source", "sink", "collector", "distributor"]:
                weight = 1.5  # Important structural positions
            else:
                weight = 1.0  # Normal weight for intermediate nodes
            
            similarities.append(sim)
            position_weights.append(weight)
        
        if not similarities:
            return 0.0
        
        # Weighted average
        return sum(s * w for s, w in zip(similarities, position_weights)) / sum(position_weights)
    
    def apply_semantic_filtering(self, topological_motifs=None):
        """
        Apply semantic filtering to group similar motifs.
        
        Args:
            topological_motifs: Dictionary of topologically grouped motifs
                               (if None, uses self.topological_motifs)
        
        Returns:
            Dictionary of semantic motif groups
        """
        if topological_motifs is None:
            topological_motifs = self.topological_motifs
        
        semantic_groups = {}
        
        # Group by motif type first (for processing efficiency)
        motif_type_groups = defaultdict(list)
        for key, motifs in topological_motifs.items():
            # Extract motif type from the key (e.g., M1, M2.1)
            motif_type = key.split('_')[0]
            motif_type_groups[motif_type].extend([(key, motif) for motif in motifs])
        
        # For each motif type, perform semantic subgrouping within isomorphism classes
        print("\nApplying semantic filtering...")
        for motif_type, motif_items in motif_type_groups.items():
            print(f"Processing {motif_type} motifs ({len(motif_items)} instances)...")
            
            # Group by size class
            size_groups = defaultdict(list)
            for key, motif in motif_items:
                size_part = '_'.join(key.split('_')[1:])  # Everything after the motif type
                size_groups[size_part].append((key, motif))
            
            # For each size class, apply semantic grouping
            for size_class, items in size_groups.items():
                if len(items) <= 1:
                    # If only one item, no need to group
                    orig_key = items[0][0]
                    semantic_groups[orig_key] = [items[0][1]]
                    continue
                
                # Extract just the motifs for semantic comparison
                motifs = [item[1] for item in items]
                orig_keys = [item[0] for item in items]
                
                # Apply semantic grouping
                processed = set()
                current_group_id = 1
                
                for i, motif1 in enumerate(motifs):
                    if i in processed:
                        continue
                        
                    # Create a new semantic group
                    group_key = f"{orig_keys[i]}_{current_group_id}"
                    semantic_groups[group_key] = [motif1]
                    processed.add(i)
                    
                    # Find semantically similar motifs
                    for j, motif2 in enumerate(motifs):
                        if j in processed or i == j:
                            continue
                        
                        # Check for isomorphism
                        matcher = nx.algorithms.isomorphism.DiGraphMatcher(motif1, motif2)
                        if matcher.is_isomorphic():
                            mapping = next(matcher.isomorphisms_iter())
                            
                            # Check semantic similarity
                            similarity = self.compute_enhanced_similarity(motif1, motif2, mapping)
                            if similarity >= self.min_semantic_similarity:
                                semantic_groups[group_key].append(motif2)
                                processed.add(j)
                    
                    # Add metadata for this group
                    self.motif_metadata[group_key] = {
                        "motif_type": motif_type,
                        "description": self.motif_types.get(motif_type, "Unknown Type"),
                        "size": len(motif1.nodes()),
                        "edges": len(motif1.edges()),
                        "instances": len(semantic_groups[group_key])
                    }
                    
                    current_group_id += 1
        
        # Update library
        self.semantic_motifs.update(semantic_groups)
        self.stats["total_semantic_groups"] = len(semantic_groups)
        
        return semantic_groups
    
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
        
        print(f"\nPerforming nearest neighbor augmentation ({num_samples} samples per group)...")
        augmented_motifs = {}
        
        # Process each semantic group
        for group_key, motifs in tqdm(self.semantic_motifs.items(), desc="Augmenting groups"):
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
                matcher = nx.algorithms.isomorphism.DiGraphMatcher(base_motif, similar_motif)
                if matcher.is_isomorphic():
                    mapping = next(matcher.isomorphisms_iter())
                    
                    # With 50% probability, borrow node labels from the similar motif
                    if random.random() < 0.5:
                        for node1, node2 in mapping.items():
                            if random.random() < 0.3:  # Only modify some nodes
                                label1 = base_motif.nodes[node1].get('label', '')
                                label2 = similar_motif.nodes[node2].get('label', '')
                                
                                # Blend the labels
                                if label1 and label2:
                                    words1 = label1.split()
                                    words2 = label2.split()
                                    if len(words1) > 1 and len(words2) > 1:
                                        # Create hybrid label by mixing words
                                        new_label = ' '.join(
                                            random.sample(words1, min(len(words1)//2, 1)) + 
                                            random.sample(words2, min(len(words2)//2, 1))
                                        )
                                        synthetic_motif.nodes[node1]['label'] = new_label
                
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
                "instances": len(new_motifs)
            }
        
        # Update library
        self.semantic_motifs.update(augmented_motifs)
        self.stats["total_augmented_motifs"] = sum(len(motifs) for key, motifs in augmented_motifs.items())
        
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
        
        print(f"\nPerforming bootstrapping augmentation ({num_samples} samples per group)...")
        augmented_motifs = {}
        
        # Collect all node labels across motifs
        all_labels = []
        for motifs in self.semantic_motifs.values():
            for motif in motifs:
                all_labels.extend([motif.nodes[n].get('label', '') for n in motif.nodes()])
        
        # Remove empty labels and duplicates while preserving order
        seen = set()
        all_labels = [label for label in all_labels if label and label not in seen and not seen.add(label)]
        
        # Process each semantic group
        for group_key, motifs in tqdm(self.semantic_motifs.items(), desc="Bootstrapping groups"):
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
                        synthetic_motif.nodes[node]['label'] = random.choice(all_labels)
                    else:
                        synthetic_motif.nodes[node]['label'] = base_motif.nodes[node].get('label', '')
                
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
                "instances": len(new_motifs)
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
            "motif_types": {k: {"description": v, "count": 0} for k, v in self.motif_types.items()},
            "semantic_groups": len(self.motif_metadata),
            "augmented_groups": sum(1 for meta in self.motif_metadata.values() if meta.get("augmented", False)),
            "total_motifs": sum(len(motifs) for motifs in self.semantic_motifs.values())
        }
        
        # Count by motif type
        for key, meta in self.motif_metadata.items():
            motif_type = meta.get("motif_type")
            if motif_type in summary["motif_types"]:
                summary["motif_types"][motif_type]["count"] += meta.get("instances", 0)
        
        return summary
    
    def visualize_motif_group(self, group_key, output_dir='output', max_examples=3):
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
        examples = motifs[:min(max_examples, len(motifs))]
        
        # Create a figure with subplots for each example
        fig, axes = plt.subplots(1, len(examples), figsize=(6 * len(examples), 5))
        if len(examples) == 1:
            axes = [axes]
        
        # Get motif type from metadata
        metadata = self.motif_metadata.get(group_key, {})
        motif_type = metadata.get("motif_type", group_key.split('_')[0])
        motif_name = metadata.get("description", self.motif_types.get(motif_type, "Unknown Type"))
        
        # Plot each example
        for i, subgraph in enumerate(examples):
            ax = axes[i]
            
            # Create node labels
            node_labels = {node: subgraph.nodes[node].get('label', str(node)) for node in subgraph.nodes()}
            
            # Simple coloring scheme based on node position in the graph
            in_degree = dict(subgraph.in_degree())
            out_degree = dict(subgraph.out_degree())
            
            node_colors = []
            for node in subgraph.nodes():
                # Source nodes (no incoming edges)
                if in_degree[node] == 0 and out_degree[node] > 0:
                    node_colors.append('skyblue')
                # Sink nodes (no outgoing edges)
                elif in_degree[node] > 0 and out_degree[node] == 0:
                    node_colors.append('salmon')
                # Intermediate nodes
                else:
                    node_colors.append('lightgreen')
            
            # Draw the graph
            pos = nx.spring_layout(subgraph, seed=42)
            nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_color=node_colors, node_size=1500, alpha=0.8)
            nx.draw_networkx_edges(subgraph, pos, ax=ax, width=2, edge_color='gray', arrowsize=20)
            nx.draw_networkx_labels(subgraph, pos, labels=node_labels, ax=ax, font_size=10)
            
            # Add a title
            if i == 0:
                ax.set_title(f"{motif_name} ({motif_type}): Group {group_key}\nSample {i+1}/{len(motifs)} examples")
            else:
                ax.set_title(f"Sample {i+1}/{len(motifs)} examples")
            
            ax.axis('off')
        
        # Save the figure
        plt.tight_layout()
        filename = f"motif_group_{group_key.replace('/', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        
        return os.path.join(output_dir, filename)
    
    def visualize_all_groups(self, output_dir='output', max_groups=None, max_examples=2):
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
            self.semantic_motifs.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        if max_groups is not None:
            sorted_groups = sorted_groups[:max_groups]
        
        # Visualize each group
        visualization_files = []
        for group_key, motifs in tqdm(sorted_groups, desc="Visualizing groups"):
            file_path = self.visualize_motif_group(
                group_key, 
                output_dir=output_dir,
                max_examples=max_examples
            )
            visualization_files.append(file_path)
        
        return visualization_files
    
    def save_library(self, file_path):
        """
        Save the motif library to a file.
        
        Args:
            file_path: Path to save the library
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a serializable version of the library
            library_data = {
                "min_motif_size": self.min_motif_size,
                "max_motif_size": self.max_motif_size,
                "min_semantic_similarity": self.min_semantic_similarity,
                "stats": self.stats,
                "motif_metadata": self.motif_metadata,
                # Convert NetworkX graphs to serializable format
                "semantic_motifs": {
                    group_key: [
                        {
                            "nodes": list(m.nodes()),
                            "edges": list(m.edges()),
                            "node_labels": {str(n): m.nodes[n].get('label', str(n)) for n in m.nodes()}
                        }
                        for m in motifs
                    ]
                    for group_key, motifs in self.semantic_motifs.items()
                }
            }
            
            # Save to JSON
            with open(file_path, 'w') as f:
                json.dump(library_data, f, indent=2)
            
            print(f"Motif library saved to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error saving motif library: {e}")
            return False
    
    @classmethod
    def load_library(cls, file_path):
        """
        Load a motif library from a file.
        
        Args:
            file_path: Path to the library file
            
        Returns:
            MotifLibrary instance
        """
        try:
            # Load JSON data
            with open(file_path, 'r') as f:
                library_data = json.load(f)
            
            # Create a new library instance
            library = cls(
                min_motif_size=library_data.get("min_motif_size", 3),
                max_motif_size=library_data.get("max_motif_size", 5),
                min_semantic_similarity=library_data.get("min_semantic_similarity", 0.4)
            )
            
            # Restore library stats and metadata
            library.stats = library_data.get("stats", {})
            library.motif_metadata = library_data.get("motif_metadata", {})
            
            # Restore semantic motifs by converting serialized data back to NetworkX graphs
            for group_key, serialized_motifs in library_data.get("semantic_motifs", {}).items():
                library.semantic_motifs[group_key] = []
                
                for motif_data in serialized_motifs:
                    G = nx.DiGraph()
                    
                    # Add nodes with labels
                    for node in motif_data.get("nodes", []):
                        G.add_node(node)
                    
                    # Add edges
                    for edge in motif_data.get("edges", []):
                        if len(edge) >= 2:
                            G.add_edge(edge[0], edge[1])
                    
                    # Add node labels
                    for node, label in motif_data.get("node_labels", {}).items():
                        if node in G.nodes():
                            G.nodes[node]['label'] = label
                    
                    library.semantic_motifs[group_key].append(G)
            
            print(f"Motif library loaded from {file_path}")
            print(f"Contains {len(library.semantic_motifs)} motif groups with {library.stats.get('total_motifs', 0)} total motifs")
            
            return library
            
        except Exception as e:
            print(f"Error loading motif library: {e}")
            return None
    
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
            summary_data = {
                "stats": self.stats,
                "motif_groups": []
            }
            
            # Add metadata for each group
            for group_key, metadata in self.motif_metadata.items():
                group_data = {
                    "group_key": group_key,
                    **metadata,
                    "instances": len(self.semantic_motifs.get(group_key, []))
                }
                summary_data["motif_groups"].append(group_data)
            
            # Save to JSON
            with open(file_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            print(f"Motif library summary exported to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting motif library summary: {e}")
            return False


# Helper functions for working with the motif library

def load_graph_from_json(file_path):
    """
    Load a causal graph from JSON file and convert to NetworkX
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        NetworkX DiGraph object
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    # Handle different JSON formats
    if "nodes" in data and isinstance(data["nodes"], dict):
        # Format with nodes and edges as dictionaries
        for node_id, node_data in data["nodes"].items():
            # Normalize the label by lowercasing and removing special characters
            raw_label = node_data.get("label", str(node_id))
            normalized_label = raw_label.lower().replace('_', ' ').strip()
            G.add_node(node_id, label=normalized_label, original_label=raw_label)
        
        # Add edges with metadata
        for edge_id, edge_data in data["edges"].items():
            G.add_edge(
                edge_data["source"], 
                edge_data["target"],
                id=edge_id,
                modifier=edge_data.get("modifier", 0),
                confidence=edge_data.get("aggregate_confidence", 0)
            )
    elif "nodes" in data and isinstance(data["nodes"], list):
        # Format with nodes and edges as arrays
        for node_data in data["nodes"]:
            node_id = node_data.get("id", str(len(G.nodes())))
            label = node_data.get("label", str(node_id))
            normalized_label = label.lower().replace('_', ' ').strip()
            G.add_node(node_id, label=normalized_label, original_label=label)
        
        # Add edges
        for edge_data in data.get("edges", []):
            source = edge_data.get("source")
            target = edge_data.get("target")
            if source and target:
                G.add_edge(
                    source, target,
                    id=edge_data.get("id", f"e{len(G.edges())}"),
                    modifier=edge_data.get("modifier", 0),
                    confidence=edge_data.get("confidence", 0)
                )
    
    # Add metadata
    G.graph["metadata"] = data.get("metadata", {})
    
    return G

def process_sample_graphs(samples_dir, output_dir=None, min_semantic_similarity=0.4):
    """
    Process all sample graphs to extract and analyze motifs
    
    Args:
        samples_dir: Directory containing sample graph JSON files
        output_dir: Directory to save output files (default: samples_dir/output)
        min_semantic_similarity: Minimum semantic similarity for grouping
        
    Returns:
        MotifLibrary instance with extracted motifs
    """
    # Default output directory
    if output_dir is None:
        output_dir = os.path.join(samples_dir, "output")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create motif library
    library = MotifLibrary(min_semantic_similarity=min_semantic_similarity)
    
    # Load all graphs
    graphs = {}
    for filename in os.listdir(samples_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(samples_dir, filename)
            sample_id = filename.replace('.json', '')
            try:
                graphs[sample_id] = load_graph_from_json(file_path)
                print(f"Loaded graph {sample_id}: {len(graphs[sample_id].nodes())} nodes, {len(graphs[sample_id].edges())} edges")
            except Exception as e:
                print(f"Error loading graph {filename}: {e}")
    
    # Process each graph
    all_topological_motifs = {}
    
    for sample_id, G in graphs.items():
        print(f"\nProcessing graph {sample_id}...")
        # Extract topological motifs
        topo_motifs = library.extract_topological_motifs(G)
        
        # Prefix keys with sample_id for uniqueness
        for key, motifs in topo_motifs.items():
            all_topological_motifs[f"{sample_id}_{key}"] = motifs
    
    # Apply semantic filtering to all motifs
    library.apply_semantic_filtering()
    
    # Augment the library
    library.augment_by_nearest_neighbor(num_samples=5)
    library.augment_by_bootstrapping(num_samples=5)
    
    # Save the library
    library.save_library(os.path.join(output_dir, "motif_library.json"))
    library.export_to_json(os.path.join(output_dir, "motif_summary.json"))
    
    # Visualize representative motifs
    library.visualize_all_groups(output_dir=os.path.join(output_dir, "motif_groups"))
    
    # Print summary
    summary = library.get_motif_summary()
    print("\nMotif library summary:")
    print(f"Total topological motifs: {summary['stats']['total_topological_motifs']}")
    print(f"Total semantic groups: {summary['semantic_groups']}")
    print(f"Total augmented groups: {summary['augmented_groups']}")
    print(f"Total motifs: {summary['total_motifs']}")
    
    for motif_type, info in summary['motif_types'].items():
        if info['count'] > 0:
            print(f"  {motif_type} ({info['description']}): {info['count']} instances")
    
    return library

def main():
    """
    Main function to run the motif library
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Motif Library Builder")
    parser.add_argument("--input", "-i", required=True, help="Input directory with graph JSON files")
    parser.add_argument("--output", "-o", help="Output directory for results")
    parser.add_argument("--similarity", "-s", type=float, default=0.4, help="Minimum semantic similarity (0-1)")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize motif groups")
    
    args = parser.parse_args()
    
    # Process sample graphs
    library = process_sample_graphs(
        args.input, 
        args.output, 
        min_semantic_similarity=args.similarity
    )
    
    # Visualize if requested
    if args.visualize and library:
        output_dir = args.output or os.path.join(args.input, "output")
        library.visualize_all_groups(output_dir=os.path.join(output_dir, "motif_groups"))
    
    print("Motif library processing complete!")

if __name__ == "__main__":
    main()