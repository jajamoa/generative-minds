"""
Semantic similarity calculation using modern language models.
"""

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    print("Please install sentence-transformers: pip install sentence-transformers")
    raise

class SemanticSimilarityEngine:
    """Semantic similarity calculation using BERT-based models"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with specified model
        
        Args:
            model_name: Name of the sentence-transformers model to use
                      Default is all-MiniLM-L6-v2 for good balance of speed/accuracy
        """
        self.model = SentenceTransformer(model_name)
        # Cache embeddings for efficiency
        self._embedding_cache = {}
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, using cache if available"""
        if text not in self._embedding_cache:
            self._embedding_cache[text] = self.model.encode(text, convert_to_tensor=True)
        return self._embedding_cache[text]
        
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts
        
        Uses cosine similarity between BERT embeddings
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
            
        # Get embeddings and move to CPU
        emb1 = self.get_embedding(text1).cpu().numpy()
        emb2 = self.get_embedding(text2).cpu().numpy()
        
        # Calculate cosine similarity
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        
    def calculate_batch_similarity(self, texts1: list, texts2: list) -> np.ndarray:
        """Calculate similarities between two lists of texts
        
        More efficient than calculating individually
        
        Args:
            texts1: First list of texts
            texts2: Second list of texts
            
        Returns:
            Matrix of similarity scores
        """
        # Get embeddings
        emb1 = self.model.encode(texts1, convert_to_tensor=True)
        emb2 = self.model.encode(texts2, convert_to_tensor=True)
        
        # Normalize
        emb1 = emb1 / np.linalg.norm(emb1, axis=1)[:, np.newaxis]
        emb2 = emb2 / np.linalg.norm(emb2, axis=1)[:, np.newaxis]
        
        # Calculate similarities
        return np.dot(emb1, emb2.T)