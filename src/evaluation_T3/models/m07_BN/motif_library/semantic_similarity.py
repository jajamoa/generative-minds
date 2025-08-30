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
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading WordNet...")
            nltk.download('wordnet')
            nltk.download('punkt')
            
        # Load word vectors if available and requested
        if use_word_vectors and os.path.exists(WORD_VECTORS_PATH):
            try:
                print(f"Loading word vectors from {WORD_VECTORS_PATH}...")
                self.word_vectors = KeyedVectors.load_word2vec_format(WORD_VECTORS_PATH, binary=True)
                print("Word vectors loaded.")
            except Exception as e:
                print(f"Error loading word vectors: {e}")
                self.use_word_vectors = False
    
    def preprocess_label(self, label):
        """Preprocess node label for semantic comparison"""
        # Convert to lowercase and remove underscores/special chars
        label = re.sub(r'[^a-zA-Z0-9\s]', ' ', label.lower())
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
    
    def motif_semantic_similarity(self, motif1, motif2, node_mapping):
        """
        Calculate semantic similarity between two motifs
        
        Args:
            motif1: First motif (NetworkX subgraph)
            motif2: Second motif (NetworkX subgraph)
            node_mapping: Dictionary mapping nodes from motif1 to motif2
        
        Returns:
            float: Similarity score between 0 and 1
        """
        # Check that the mapping is valid
        if not node_mapping or len(node_mapping) == 0:
            return 0.0
        
        # Calculate node-by-node similarity and take average
        similarities = []
        for node1, node2 in node_mapping.items():
            label1 = motif1.nodes[node1].get('label', node1)
            label2 = motif2.nodes[node2].get('label', node2)
            sim = self.node_similarity(label1, label2)
            similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        return sum(similarities) / len(similarities)

def identify_topological_motifs(graph):
    """
    Identify the three basic motif types in the graph
    
    Args:
        graph: NetworkX DiGraph of the entire causal graph
        
    Returns:
        Dictionary of topologically classified motifs
    """
    motifs = {
        "M1_Chain": [],  # A → B → C
        "M2_Fork": [],   # A → B, A → C
        "M3_Collider": [] # A → B ← C
    }
    
    # Implementation for identifying basic motifs
    # For demonstration - actual implementation would involve
    # subgraph enumeration and pattern matching
    
    return motifs

def apply_semantic_grouping(topological_motifs, min_similarity=0.7):
    """
    Group topologically-similar motifs by semantic similarity
    
    Args:
        topological_motifs: Dictionary of topologically classified motifs
        min_similarity: Minimum semantic similarity to group motifs (0-1)
    
    Returns:
        Dictionary of semantic motif groups
    """
    similarity_engine = SemanticSimilarityEngine()
    semantic_groups = {}
    
    # For each topological category, perform semantic subgrouping
    for topo_type, motifs in topological_motifs.items():
        # Skip if there's only one motif in the group
        if len(motifs) <= 1:
            semantic_groups[f"{topo_type}_1"] = motifs
            continue
        
        # Compare all pairs within this topological group
        processed = set()
        current_group_id = 1
        
        for i, motif1 in enumerate(motifs):
            if i in processed:
                continue
                
            # Create a new semantic group
            group_key = f"{topo_type}_{current_group_id}"
            semantic_groups[group_key] = [motif1]
            processed.add(i)
            
            # Find semantically similar motifs
            for j, motif2 in enumerate(motifs):
                if j in processed or i == j:
                    continue
                
                # Get the isomorphism mapping
                matcher = nx.algorithms.isomorphism.DiGraphMatcher(motif1, motif2)
                if matcher.is_isomorphic():
                    mapping = next(matcher.isomorphisms_iter())
                    
                    # Check semantic similarity
                    similarity = similarity_engine.motif_semantic_similarity(motif1, motif2, mapping)
                    if similarity >= min_similarity:
                        semantic_groups[group_key].append(motif2)
                        processed.add(j)
            
            current_group_id += 1
    
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