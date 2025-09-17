#!/usr/bin/env python3
"""
Node similarity computation for CBN construction.
Provides semantic similarity matching for node merging using embedding models.
"""

import hashlib
import re
from typing import Dict, List, Tuple, Optional, Union
from functools import lru_cache
import numpy as np

# Embedding model imports with fallbacks
_embedding_model = None
_embedding_type = None

def _initialize_embedding_model():
    """Initialize the best available embedding model."""
    global _embedding_model, _embedding_type
    
    if _embedding_model is not None:
        return
    
    # Try sentence-transformers first (best option)
    try:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, good quality
        _embedding_type = 'sentence_transformer'
        print("Using sentence-transformers for semantic similarity")
        return
    except ImportError:
        pass
    
    # Try OpenAI embeddings as fallback
    try:
        import openai
        import os
        if os.getenv('OPENAI_API_KEY'):
            _embedding_model = "text-embedding-3-small"  # Cheaper option
            _embedding_type = 'openai'
            print("Using OpenAI embeddings for semantic similarity")
            return
    except ImportError:
        pass
    
    # Fallback to word-based similarity
    _embedding_type = 'word_based'
    print("Warning: No embedding models available, using word-based similarity fallback")

def _get_embedding(text: str) -> np.ndarray:
    """Get embedding for a text using the available model."""
    global _embedding_model, _embedding_type
    
    if _embedding_type == 'sentence_transformer':
        return _embedding_model.encode([text.strip()])[0]
    
    elif _embedding_type == 'openai':
        import openai
        try:
            response = openai.embeddings.create(
                model=_embedding_model,
                input=text.strip()
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"OpenAI embedding error: {e}, falling back to word similarity")
            return _word_based_embedding(text)
    
    else:  # word_based fallback
        return _word_based_embedding(text)

def _word_based_embedding(text: str) -> np.ndarray:
    """Fallback word-based pseudo-embedding."""
    # Simple bag-of-words style representation
    words = re.findall(r'\w+', text.lower())
    
    # Create a hash-based feature vector
    feature_size = 100
    embedding = np.zeros(feature_size)
    
    for word in words:
        word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
        indices = [word_hash % feature_size, (word_hash // feature_size) % feature_size]
        for idx in indices:
            embedding[idx] += 1.0
    
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding

def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    if norms == 0:
        return 0.0
    
    return float(dot_product / norms)

@lru_cache(maxsize=1000)
def compute_node_similarity(label1: str, label2: str) -> float:
    """
    Compute semantic similarity between two node labels using embeddings.
    
    Args:
        label1: First node label
        label2: Second node label
        
    Returns:
        Similarity score between 0 and 1
    """
    if not label1 or not label2:
        return 0.0
    
    # Exact match
    if label1.strip().lower() == label2.strip().lower():
        return 1.0
    
    # Initialize embedding model if not done
    _initialize_embedding_model()
    
    try:
        # Get embeddings
        emb1 = _get_embedding(label1)
        emb2 = _get_embedding(label2)
        
        # Compute cosine similarity
        similarity = _cosine_similarity(emb1, emb2)
        
        # Ensure similarity is in [0, 1] range
        similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
        
        return similarity
        
    except Exception as e:
        print(f"Error computing similarity for '{label1}' and '{label2}': {e}")
        # Fallback to simple string match
        return 1.0 if label1.lower() == label2.lower() else 0.0

def find_similar_node(target_label: str, existing_nodes: Dict[str, Dict], 
                     threshold: float = 0.8) -> Tuple[Optional[str], float]:
    """
    Find the most similar existing node to target_label.
    
    Args:
        target_label: Label to find similarity for
        existing_nodes: Dictionary of existing nodes {node_id: node_data}
        threshold: Minimum similarity threshold
        
    Returns:
        Tuple of (Node ID of most similar node or None, similarity score)
    """
    best_similarity = 0.0
    best_node_id = None
    
    for node_id, node_data in existing_nodes.items():
        existing_label = node_data.get("label", "")
        similarity = compute_node_similarity(target_label, existing_label)
        
        if similarity >= threshold and similarity > best_similarity:
            best_similarity = similarity
            best_node_id = node_id
    
    return best_node_id, best_similarity

def get_consistent_node_id(label: str, existing_nodes: Dict[str, Dict], 
                          similarity_threshold: float = 0.8) -> str:
    """
    Get a consistent node ID, either by finding a similar existing node
    or creating a new one.
    
    Args:
        label: Node label
        existing_nodes: Dictionary of existing nodes
        similarity_threshold: Threshold for considering nodes similar
        
    Returns:
        Node ID (either existing or new)
    """
    # Try to find similar existing node
    similar_node_id, similarity = find_similar_node(label, existing_nodes, similarity_threshold)
    
    if similar_node_id:
        return similar_node_id
    
    # Create new node ID
    hash_val = hashlib.md5(label.encode()).hexdigest()[:8]
    return f"n_{hash_val}"

def is_stance_node(label: str) -> bool:
    """
    Determine if a node represents a final stance/belief using semantic patterns.
    Stance nodes should be endpoints in the belief network.
    """
    label_lower = label.lower()
    
    # Strong indicators of stance nodes (minimal rule-based for core patterns)
    stance_indicators = [
        'support for',
        'opposition to', 
        'support',
        'oppose',
        'favor',
        'against'
    ]
    
    # Check direct indicators
    for indicator in stance_indicators:
        if indicator in label_lower:
            return True
    
    return False

def compute_similarity_matrix(labels: List[str]) -> np.ndarray:
    """
    Compute pairwise similarity matrix for a list of labels.
    Useful for clustering and analysis.
    
    Args:
        labels: List of node labels
        
    Returns:
        Square similarity matrix
    """
    n = len(labels)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                similarity = compute_node_similarity(labels[i], labels[j])
                matrix[i, j] = similarity
                matrix[j, i] = similarity  # Symmetric
    
    return matrix
