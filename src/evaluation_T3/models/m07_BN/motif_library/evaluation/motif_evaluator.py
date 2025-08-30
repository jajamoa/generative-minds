"""
A tool for evaluating the quality of Motif Library.

Evaluates:
1. Coverage - Semantic concept coverage from original graphs
2. Coherence - Node label distinctness within motifs
3. Diversity - Redundancy between motifs using clustering and entropy
"""

import os
import json
try:
    import numpy as np
    import networkx as nx
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    from tqdm import tqdm
except ImportError:
    print("Please install required packages: numpy, networkx, scipy, tqdm")
    raise

from collections import defaultdict
from typing import Dict, List, Set, Tuple

class MotifEvaluator:
    """Evaluates quality metrics for a Motif Library"""
    
    # Define motif type descriptions
    MOTIF_TYPES = {
        'M1': 'Chain',
        'M2.1': 'Basic Fork (1-to-2)',
        'M2.2': 'Extended Fork (1-to-3)',
        'M2.3': 'Large Fork (1-to-4+)',
        'M3.1': 'Basic Collider (2-to-1)',
        'M3.2': 'Extended Collider (3-to-1)',
        'M3.3': 'Large Collider (4+-to-1)'
    }
    
    def __init__(self, motif_lib_path: str):
        """Initialize evaluator
        
        Args:
            motif_lib_path: Path to motif library JSON file
        """
        self.motif_lib = self._load_motif_library(motif_lib_path)
        # Initialize semantic similarity engine
        from semantic_similarity import SemanticSimilarityEngine
        self.sim_engine = SemanticSimilarityEngine()
        
    def _load_motif_library(self, path: str) -> dict:
        """Load motif library from JSON file"""
        with open(path) as f:
            return json.load(f)
            
    def evaluate_coverage(self, agents_file: str) -> Dict[str, float]:
        """Evaluate semantic concept coverage against original graphs
        
        Args:
            agents_file: Path to agents_with_graphs.json file
            
        Returns:
            Dictionary with coverage scores
        """
        print("\nEvaluating coverage...")
        
        # Extract concepts from original graphs
        print("1. Extracting concepts from original graphs...")
        orig_concepts = self._extract_semantic_concepts(agents_file)
        
        # Extract concepts from motif library
        print("2. Extracting concepts from motif library...")
        lib_concepts = self._extract_library_concepts()
        
        # Calculate coverage scores
        print("3. Calculating coverage scores...")
        
        # Split concepts into nodes and edges
        orig_nodes = {k: v for k, v in orig_concepts.items() if '_' not in k}
        orig_edges = {k: v for k, v in orig_concepts.items() if '_' in k}
        lib_nodes = {k: v for k, v in lib_concepts.items() if '_' not in k}
        lib_edges = {k: v for k, v in lib_concepts.items() if '_' in k}
        
        # Calculate separate scores
        node_coverage = self._calculate_semantic_overlap(orig_nodes, lib_nodes)
        edge_coverage = self._calculate_semantic_overlap(orig_edges, lib_edges)
        semantic_coverage = self._calculate_semantic_overlap(orig_concepts, lib_concepts)
        
        return {
            'node_coverage': node_coverage,
            'edge_coverage': edge_coverage,
            'semantic_coverage': semantic_coverage
        }
        
    def evaluate_coherence(self) -> float:
        """Evaluate node label distinctness within motifs
        
        Lower score means more distinct labels within motifs,
        which is generally better for capturing meaningful patterns
        
        Returns:
            Coherence score (0-1), lower is better
        """
        print("\nEvaluating coherence...")
        coherence_scores = []
        
        groups = list(self.motif_lib.get("semantic_motifs", {}).values())
        for group in tqdm(groups, desc="Processing motif groups"):
            for motif in group:
                labels = list(motif.get("node_labels", {}).values())
                if len(labels) > 1:
                    # Calculate pairwise similarities
                    sims = []
                    for i in range(len(labels)):
                        for j in range(i + 1, len(labels)):
                            sim = self._label_similarity(labels[i], labels[j])
                            sims.append(sim)
                    # Higher average similarity means less distinct labels
                    coherence_scores.append(np.mean(sims) if sims else 0.0)
                
        return np.mean(coherence_scores) if coherence_scores else 0.0
        
    def evaluate_diversity(self) -> Dict[str, float]:
        """Evaluate diversity of motifs using semantic and structural metrics
        
        Metrics:
        1. Semantic diversity - Using BERT embeddings to measure semantic distinctness
        2. Structural diversity - Using graph features to measure structural distinctness
        3. Distribution entropy - Evaluating pattern distribution uniformity
        
        Returns:
            Dictionary with diversity scores
        """
        print("\nEvaluating diversity...")
        motifs = self.motif_lib.get("semantic_motifs", {})
        if not motifs:
            return {'semantic_diversity': 0.0, 'structural_diversity': 0.0, 'distribution_entropy': 0.0}
            
        # Analyze motif type distribution
        print("\nMotif type distribution:")
        
        # Detailed type counts
        type_counts = defaultdict(int)
        for group_key in motifs.keys():
            motif_type = group_key.split('_')[0]  # e.g., M1, M2.1, etc.
            type_counts[motif_type] += len(motifs[group_key])
        
        total = sum(type_counts.values())
        
        # Group by basic patterns
        basic_patterns = {
            'Chain': sum(count for mtype, count in type_counts.items() if mtype.startswith('M1')),
            'Fork': sum(count for mtype, count in type_counts.items() if mtype.startswith('M2')),
            'Collider': sum(count for mtype, count in type_counts.items() if mtype.startswith('M3'))
        }
        
        # Print basic pattern distribution
        print("\nBasic Pattern Distribution:")
        for pattern, count in basic_patterns.items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {pattern:8}: {count:4d} motifs ({percentage:5.1f}%)")
            
        # Print detailed type distribution
        print("\nDetailed Type Distribution:")
        for mtype, count in sorted(type_counts.items()):
            percentage = (count / total) * 100 if total > 0 else 0
            pattern_type = self.MOTIF_TYPES.get(mtype, mtype)
            print(f"  {mtype:4}: {count:4d} motifs ({percentage:5.1f}%) - {pattern_type}")
            
        print("\nCalculating diversity metrics...")
        
        # 1. Semantic diversity
        print("1. Calculating semantic diversity...")
        semantic_div = self._calculate_semantic_diversity(motifs)
        
        # 2. Structural diversity
        print("2. Calculating structural diversity...")
        structural_div = self._calculate_structural_diversity(motifs)
        
        # 3. Distribution entropy
        print("3. Calculating distribution entropy...")
        entropy = self._calculate_entropy(motifs)
        
        return {
            'semantic_diversity': semantic_div,
            'structural_diversity': structural_div,
            'distribution_entropy': entropy
        }
        
    def _extract_semantic_concepts(self, agents_file: str) -> Dict[str, float]:
        """Extract semantic concepts from agents_with_graphs.json"""
        concepts = defaultdict(float)
        total_nodes = 0
        
        with open(agents_file) as f:
            agents = json.load(f)
            
        for agent in tqdm(agents, desc="Processing agents"):
            # Process each graph from the agent
            for g in agent.get("graphs", []) or []:
                graph = g.get("graph", {})
                
                # Extract node concepts
                nodes = graph.get("nodes", {})
                for node_data in nodes.values():
                    label = node_data.get("label", "").lower()
                    if label:
                        words = label.split()
                        for word in words:
                            concepts[word] += 1
                            total_nodes += 1
                            
                # Extract relationship concepts
                edges = graph.get("edges", {})
                for edge_data in edges.values():
                    source = edge_data.get("source")
                    target = edge_data.get("target")
                    if source and target:
                        source_label = nodes.get(source, {}).get("label", "").lower()
                        target_label = nodes.get(target, {}).get("label", "").lower()
                        if source_label and target_label:
                            rel = f"{source_label}_{target_label}"
                            concepts[rel] += 1
                            
        # Normalize frequencies
        if total_nodes > 0:
            for concept in concepts:
                concepts[concept] /= total_nodes
                
        return dict(concepts)
        
    def _extract_library_concepts(self) -> Dict[str, float]:
        """Extract semantic concepts from motif library"""
        concepts = defaultdict(float)
        total_nodes = 0
        
        for group in tqdm(self.motif_lib.get("semantic_motifs", {}).values(), desc="Processing motif groups"):
            for motif in group:
                # Node concepts
                for label in motif.get("node_labels", {}).values():
                    label = label.lower()
                    words = label.split()
                    for word in words:
                        concepts[word] += 1
                        total_nodes += 1
                
                # Relationship concepts
                edges = motif.get("edges", [])
                labels = motif.get("node_labels", {})
                for edge in edges:
                    if len(edge) >= 2:
                        source_label = labels.get(str(edge[0]), "").lower()
                        target_label = labels.get(str(edge[1]), "").lower()
                        if source_label and target_label:
                            rel = f"{source_label}_{target_label}"
                            concepts[rel] += 1
                            
        # Normalize frequencies
        if total_nodes > 0:
            for concept in concepts:
                concepts[concept] /= total_nodes
                
        return dict(concepts)
        
    def _calculate_semantic_overlap(self, orig_concepts: Dict[str, float], 
                                  lib_concepts: Dict[str, float]) -> float:
        """Calculate semantic overlap using cosine similarity"""
        # Get all unique concepts
        all_concepts = set(orig_concepts) | set(lib_concepts)
        
        # Convert to vectors
        orig_vec = np.array([orig_concepts.get(c, 0) for c in all_concepts])
        lib_vec = np.array([lib_concepts.get(c, 0) for c in all_concepts])
        
        # Calculate cosine similarity
        norm_orig = np.linalg.norm(orig_vec)
        norm_lib = np.linalg.norm(lib_vec)
        if norm_orig == 0 or norm_lib == 0:
            return 0.0
            
        return np.dot(orig_vec, lib_vec) / (norm_orig * norm_lib)
        
    def _label_similarity(self, label1: str, label2: str) -> float:
        """Calculate label similarity using BERT embeddings"""
        return self.sim_engine.calculate_similarity(label1, label2)
        
    def _calculate_semantic_diversity(self, motifs: dict) -> float:
        """Calculate semantic diversity between motif groups using BERT embeddings
        
        Higher score means more semantically distinct groups
        """
        # Get representative text for each motif group
        group_texts = []
        for group in motifs.values():
            # Combine all node labels in the group
            all_labels = []
            for motif in group:
                all_labels.extend(motif.get("node_labels", {}).values())
            group_text = " ".join(all_labels)
            if group_text.strip():
                group_texts.append(group_text)
                
        if len(group_texts) < 2:
            return 0.0
            
        # Calculate pairwise similarities
        similarities = []
        for i in tqdm(range(len(group_texts)), desc="Calculating group similarities"):
            for j in range(i + 1, len(group_texts)):
                sim = self.sim_engine.calculate_similarity(group_texts[i], group_texts[j])
                similarities.append(sim)
                
        # Convert similarities to diversity score (1 - average similarity)
        avg_sim = np.mean(similarities) if similarities else 0.0
        return 1.0 - avg_sim
        
    def _calculate_structural_diversity(self, motifs: dict) -> float:
        """Calculate structural diversity between motif groups
        
        Uses graph features like node count, edge count, degree distributions
        Higher score means more structurally distinct groups
        """
        features = []
        for group in tqdm(motifs.values(), desc="Extracting structural features"):
            # Calculate average structural features for the group
            nodes = []
            edges = []
            avg_degrees = []
            for motif in group:
                n = len(motif.get("nodes", []))
                e = len(motif.get("edges", []))
                nodes.append(n)
                edges.append(e)
                avg_degrees.append(2 * e / n if n > 0 else 0)
                
            group_vec = [
                np.mean(nodes),
                np.mean(edges),
                np.mean(avg_degrees),
                np.std(nodes),
                np.std(edges),
                np.std(avg_degrees)
            ]
            features.append(group_vec)
            
        if len(features) < 2:
            return 0.0
            
        # Calculate average pairwise distances
        X = np.array(features)
        distances = pdist(X, metric='euclidean')
        
        # Normalize by maximum possible distance
        max_dist = np.sqrt(X.shape[1])  # Maximum possible Euclidean distance for normalized features
        return np.mean(distances) / max_dist
        
    def _calculate_entropy(self, motifs: dict) -> float:
        """Calculate information entropy of motif distribution"""
        counts = [len(group) for group in motifs.values()]
        total = sum(counts)
        
        if not total:
            return 0.0
            
        probs = [c/total for c in counts]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return entropy / np.log2(len(counts)) if len(counts) > 1 else 1.0

def main():
    """Main function for command line execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Motif Library quality')
    parser.add_argument('--lib', required=True, help='Path to motif library JSON file')
    parser.add_argument('--orig', required=True, help='Path to agents_with_graphs.json file')
    
    args = parser.parse_args()
    
    evaluator = MotifEvaluator(args.lib)
    
    # Run evaluation
    coverage_scores = evaluator.evaluate_coverage(args.orig)
    coherence = evaluator.evaluate_coherence()
    diversity_scores = evaluator.evaluate_diversity()
    
    # Print results
    print("\nEvaluation Results:")
    print("\nCoverage Scores:")
    print(f"  - Node Coverage: {coverage_scores['node_coverage']:.3f}")
    print(f"  - Edge Coverage: {coverage_scores['edge_coverage']:.3f}")
    print(f"  - Overall Semantic Coverage: {coverage_scores['semantic_coverage']:.3f}")
    
    print("\nCoherence Score:")
    print(f"  {coherence:.3f} (lower means more distinct labels within motifs)")
    
    print("\nDiversity Scores:")
    print(f"  - Semantic Diversity: {diversity_scores['semantic_diversity']:.3f}")
    print(f"  - Structural Diversity: {diversity_scores['structural_diversity']:.3f}")
    print(f"  - Distribution Entropy: {diversity_scores['distribution_entropy']:.3f}")

if __name__ == '__main__':
    main()