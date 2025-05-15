#!/usr/bin/env python3
"""
Semantic similarity and filtering for motif analysis

This module provides tools for comparing the semantic similarity of nodes
and motifs after topology-based matching has been performed.
"""

import os
import json
import numpy as np
from collections import defaultdict
import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from gensim.models import KeyedVectors
import re

# Path to pretrained word vectors - optional and only used if use_word_vectors=True
# For example: /path/to/GoogleNews-vectors-negative300.bin or similar
WORD_VECTORS_PATH = None  # Set to None by default, specify a path if needed


class SemanticSimilarityEngine:
    """
    Engine for computing semantic similarity between nodes and motifs
    """

    def __init__(self, use_wordnet=True, use_word_vectors=False):
        """Initialize the semantic similarity engine"""
        self.use_wordnet = use_wordnet
        self.use_word_vectors = use_word_vectors
        self.word_vectors = None

        # Prepare NLTK resources
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            print("Downloading WordNet...")
            nltk.download("wordnet")
            nltk.download("punkt")

        # Load word vectors if available and requested
        if use_word_vectors and os.path.exists(WORD_VECTORS_PATH):
            try:
                print(f"Loading word vectors from {WORD_VECTORS_PATH}...")
                self.word_vectors = KeyedVectors.load_word2vec_format(
                    WORD_VECTORS_PATH, binary=True
                )
                print("Word vectors loaded.")
            except Exception as e:
                print(f"Error loading word vectors: {e}")
                self.use_word_vectors = False

        # Add position weights as specified in the paper
        self.position_weights = {
            "source": 1.5,  # Higher weight for source nodes
            "sink": 1.5,  # Higher weight for sink nodes
            "intermediate": 1.0,  # Base weight for intermediate nodes
        }

    def preprocess_label(self, label):
        """Preprocess node label for semantic comparison"""
        # Convert to lowercase and remove underscores/special chars
        label = re.sub(r"[^a-zA-Z0-9\s]", " ", label.lower())
        # Split into words
        return label.split()

    def node_similarity(self, label1, label2):
        """
        Calculate semantic similarity between two node labels

        Returns:
            float: Similarity score between 0 and 1
        """
        # Preprocess labels
        words1 = self.preprocess_label(label1)
        words2 = self.preprocess_label(label2)

        # If WordNet is available, use it for similarity
        if self.use_wordnet:
            # Calculate average pairwise similarity
            total_sim = 0.0
            count = 0
            for w1 in words1:
                for w2 in words2:
                    sim = self.wordnet_similarity(w1, w2)
                    if sim > 0:
                        total_sim += sim
                        count += 1

            if count > 0:
                return total_sim / count

        # Simple token overlap (fallback)
        common_words = set(words1).intersection(set(words2))
        total_words = set(words1).union(set(words2))

        if not total_words:
            return 0.0

        return len(common_words) / len(total_words)

    def wordnet_similarity(self, word1, word2):
        """Calculate semantic similarity between words using WordNet"""
        try:
            # Get all synsets for both words
            synsets1 = wordnet.synsets(word1)
            synsets2 = wordnet.synsets(word2)

            if not synsets1 or not synsets2:
                return 0.0

            # Find the maximum similarity between any pair of synsets
            max_sim = 0.0
            for s1 in synsets1:
                for s2 in synsets2:
                    try:
                        sim = s1.path_similarity(s2)
                        if sim and sim > max_sim:
                            max_sim = sim
                    except:
                        continue
            return max_sim
        except:
            return 0.0

    def word_vector_similarity(self, word1, word2):
        """Calculate semantic similarity using word vectors"""
        if not self.word_vectors:
            return 0.0

        try:
            if word1 in self.word_vectors and word2 in self.word_vectors:
                return self.word_vectors.similarity(word1, word2)
        except:
            pass

        return 0.0

    def get_node_position(self, G, node):
        """Determine node's structural position in the motif"""
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)

        if in_degree == 0:
            return "source"
        elif out_degree == 0:
            return "sink"
        else:
            return "intermediate"

    def motif_semantic_similarity(self, motif1, motif2, node_mapping):
        """
        Calculate position-weighted semantic similarity between two motifs
        using the formula from the paper:
        Sim(m1, m2) = Σ(w_i * s_i) / Σ(w_i)
        """
        if not node_mapping or len(node_mapping) == 0:
            return 0.0

        total_weighted_sim = 0.0
        total_weight = 0.0

        for node1, node2 in node_mapping.items():
            # Get node labels
            label1 = motif1.nodes[node1].get("label", node1)
            label2 = motif2.nodes[node2].get("label", node2)

            # Calculate base semantic similarity
            sim = self.node_similarity(label1, label2)

            # Get position-based weights
            pos1 = self.get_node_position(motif1, node1)
            pos2 = self.get_node_position(motif2, node2)

            # Use average weight if positions differ
            weight = (self.position_weights[pos1] + self.position_weights[pos2]) / 2

            total_weighted_sim += weight * sim
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_weighted_sim / total_weight


def identify_motif_type(G):
    """
    Identify detailed motif type based on the paper's classification
    """
    # Get basic graph properties
    nodes = list(G.nodes())
    n = len(nodes)

    if n < 3:
        return "Unknown"

    # Find sources and sinks
    sources = [n for n in nodes if G.in_degree(n) == 0]
    sinks = [n for n in nodes if G.out_degree(n) == 0]

    # Chain (M1) - exactly 3 nodes with linear sequence
    if n == 3 and len(sources) == 1 and len(sinks) == 1:
        # Verify it's a linear chain
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


def apply_semantic_grouping(topological_motifs, min_similarity=0.7):
    """
    Group motifs by semantic similarity within each topological class
    """
    similarity_engine = SemanticSimilarityEngine()
    semantic_groups = defaultdict(list)

    for motif in topological_motifs:
        # Get detailed motif type
        motif_type = identify_motif_type(motif)

        # Find best matching group or create new one
        best_match = None
        best_sim = 0

        for group in semantic_groups[motif_type]:
            # Compare with representative motif of the group
            rep_motif = group[0]
            # Get node mapping (implement this based on your graph structure)
            node_mapping = get_node_mapping(motif, rep_motif)
            sim = similarity_engine.motif_semantic_similarity(
                motif, rep_motif, node_mapping
            )

            if sim > min_similarity and sim > best_sim:
                best_match = group
                best_sim = sim

        if best_match is not None:
            best_match.append(motif)
        else:
            semantic_groups[motif_type].append([motif])

    return semantic_groups


def analyze_motifs(graph, min_similarity=0.7):
    """
    Complete motif analysis pipeline

    Args:
        graph: NetworkX DiGraph of the entire causal graph
        min_similarity: Minimum semantic similarity threshold

    Returns:
        Dictionary of semantic motif groups
    """
    # Step 1: Topological classification
    topological_motifs = identify_topological_motifs(graph)

    # Step 2: Semantic subgrouping within each topological class
    semantic_groups = apply_semantic_grouping(topological_motifs, min_similarity)

    return semantic_groups


def main():
    """Test semantic similarity functionality"""
    print("Testing semantic similarity module...")

    # Create a semantic similarity engine
    engine = SemanticSimilarityEngine()

    # Test some node label similarities
    test_pairs = [
        ("transit_access", "mobility_options"),
        ("housing_affordability", "rental_cost"),
        ("building_height", "construction_density"),
    ]

    for label1, label2 in test_pairs:
        sim = engine.node_similarity(label1, label2)
        print(f"Similarity between '{label1}' and '{label2}': {sim:.3f}")

    print("\nSemantic similarity module test complete.")


if __name__ == "__main__":
    main()
