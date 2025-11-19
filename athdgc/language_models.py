"""
Language Model Integration for Diachronic Semantics
Text embeddings and semantic similarity for historical texts
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LanguageModelIntegration:
    """Language model tools for diachronic semantic analysis"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._init_model()
    
    def _init_model(self):
        """Initialize sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("✓ Sentence transformer model loaded")
        except ImportError:
            logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for list of texts"""
        if self.model is None:
            return None
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            logger.info(f"✓ Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> Optional[float]:
        """Calculate semantic similarity between two texts"""
        if self.model is None:
            # Fallback to simple word overlap
            return self._simple_similarity(text1, text2)
        
        try:
            embeddings = self.model.encode([text1, text2])
            
            # Cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return self._simple_similarity(text1, text2)
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity fallback"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def analyze_diachronic_semantics(self, period_a_texts: List[str], 
                                     period_b_texts: List[str],
                                     target_word: Optional[str] = None) -> Dict:
        """
        Analyze semantic shift between two periods
        
        Args:
            period_a_texts: Texts from period A
            period_b_texts: Texts from period B
            target_word: Optional target word to focus analysis
        
        Returns:
            Semantic shift analysis
        """
        if self.model is None:
            return {'error': 'Language model not available'}
        
        # Filter texts containing target word if specified
        if target_word:
            period_a_texts = [t for t in period_a_texts if target_word.lower() in t.lower()]
            period_b_texts = [t for t in period_b_texts if target_word.lower() in t.lower()]
        
        if not period_a_texts or not period_b_texts:
            return {'error': 'Insufficient texts for analysis'}
        
        # Generate embeddings
        embeddings_a = self.generate_embeddings(period_a_texts)
        embeddings_b = self.generate_embeddings(period_b_texts)
        
        if embeddings_a is None or embeddings_b is None:
            return {'error': 'Could not generate embeddings'}
        
        # Calculate centroids
        centroid_a = np.mean(embeddings_a, axis=0)
        centroid_b = np.mean(embeddings_b, axis=0)
        
        # Calculate shift magnitude
        shift_magnitude = np.linalg.norm(centroid_a - centroid_b)
        
        # Calculate cosine similarity
        centroid_similarity = np.dot(centroid_a, centroid_b) / (
            np.linalg.norm(centroid_a) * np.linalg.norm(centroid_b)
        )
        
        # Calculate intra-period variance
        variance_a = np.mean([np.linalg.norm(emb - centroid_a) for emb in embeddings_a])
        variance_b = np.mean([np.linalg.norm(emb - centroid_b) for emb in embeddings_b])
        
        result = {
            'target_word': target_word,
            'period_a_samples': len(period_a_texts),
            'period_b_samples': len(period_b_texts),
            'shift_magnitude': float(shift_magnitude),
            'centroid_similarity': float(centroid_similarity),
            'variance_a': float(variance_a),
            'variance_b': float(variance_b),
            'semantic_shift_detected': shift_magnitude > 0.5,
            'shift_interpretation': self._interpret_shift(shift_magnitude, centroid_similarity)
        }
        
        logger.info(f"Diachronic semantic analysis: shift={shift_magnitude:.3f}")
        return result
    
    def _interpret_shift(self, magnitude: float, similarity: float) -> str:
        """Interpret semantic shift results"""
        if magnitude < 0.3:
            return 'stable - minimal semantic change'
        elif magnitude < 0.7:
            if similarity > 0.7:
                return 'moderate - related meanings with drift'
            else:
                return 'moderate - noticeable semantic change'
        else:
            if similarity > 0.5:
                return 'significant - substantial but related change'
            else:
                return 'major - radical semantic shift'
    
    def find_nearest_neighbors(self, query_text: str, candidate_texts: List[str], 
                              top_k: int = 5) -> List[Tuple[str, float]]:
        """Find semantically similar texts"""
        if self.model is None:
            return []
        
        try:
            # Generate embeddings
            query_embedding = self.model.encode([query_text])[0]
            candidate_embeddings = self.model.encode(candidate_texts)
            
            # Calculate similarities
            similarities = []
            for i, candidate_emb in enumerate(candidate_embeddings):
                sim = np.dot(query_embedding, candidate_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(candidate_emb)
                )
                similarities.append((candidate_texts[i], float(sim)))
            
            # Sort and return top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding neighbors: {e}")
            return []
    
    def cluster_texts(self, texts: List[str], n_clusters: int = 3) -> Dict:
        """Cluster texts by semantic similarity"""
        if self.model is None:
            return {'error': 'Language model not available'}
        
        try:
            from sklearn.cluster import KMeans
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            if embeddings is None:
                return {'error': 'Could not generate embeddings'}
            
            # Cluster
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            
            # Organize results
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(texts[i])
            
            return {
                'n_clusters': n_clusters,
                'clusters': {int(k): v for k, v in clusters.items()},
                'cluster_sizes': {int(k): len(v) for k, v in clusters.items()}
            }
            
        except ImportError:
            return {'error': 'scikit-learn not installed'}
        except Exception as e:
            logger.error(f"Error clustering: {e}")
            return {'error': str(e)}
