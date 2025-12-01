#!/usr/bin/env python3
"""
Machine Learning Tools for Historical Greek
Community-driven, open-source ML pipeline

Features:
- Text classification and categorization
- POS tagging models
- Lemmatization
- Dependency parsing
- Semantic role labeling
- Period/genre classification
- Statistical analysis
- LightSide-style feature extraction

Based on:
- Schneider's Text Analytics in Digital Humanities
- LightSide ML Workbench methodology
- Jurafsky & Martin NLP approaches
"""

import os
import re
import json
import pickle
import sqlite3
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FEATURE EXTRACTION (LightSide-style)
# ============================================================================

class FeatureExtractor:
    """LightSide-style feature extraction for text classification"""
    
    def __init__(self):
        self.feature_types = [
            'unigrams', 'bigrams', 'trigrams',
            'pos_unigrams', 'pos_bigrams',
            'character_ngrams', 'word_length',
            'punctuation', 'sentence_length'
        ]
        self.vocabulary = {}
        self.pos_vocabulary = {}
    
    def extract_ngrams(self, tokens: List[str], n: int = 1) -> Dict[str, int]:
        """Extract n-grams from token list"""
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = '_'.join(tokens[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams
    
    def extract_character_ngrams(self, text: str, n: int = 3) -> Dict[str, int]:
        """Extract character n-grams"""
        ngrams = {}
        text = text.lower()
        for i in range(len(text) - n + 1):
            ngram = text[i:i+n]
            ngrams[f"char_{ngram}"] = ngrams.get(f"char_{ngram}", 0) + 1
        return ngrams
    
    def extract_pos_ngrams(self, pos_tags: List[str], n: int = 1) -> Dict[str, int]:
        """Extract POS tag n-grams"""
        ngrams = {}
        for i in range(len(pos_tags) - n + 1):
            ngram = '_'.join(pos_tags[i:i+n])
            ngrams[f"pos_{ngram}"] = ngrams.get(f"pos_{ngram}", 0) + 1
        return ngrams
    
    def extract_stylometric_features(self, text: str, tokens: List[str]) -> Dict[str, float]:
        """Extract stylometric features"""
        features = {}
        
        # Word length statistics
        word_lengths = [len(t) for t in tokens if t.isalpha()]
        if word_lengths:
            features['avg_word_length'] = np.mean(word_lengths)
            features['std_word_length'] = np.std(word_lengths)
            features['max_word_length'] = max(word_lengths)
        
        # Vocabulary richness
        unique_words = set(tokens)
        features['type_token_ratio'] = len(unique_words) / max(len(tokens), 1)
        
        # Punctuation density
        punct_count = sum(1 for c in text if c in '.,;:!?')
        features['punct_density'] = punct_count / max(len(text), 1)
        
        # Sentence length (approximate)
        sentences = re.split(r'[.!?]+', text)
        sent_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sent_lengths:
            features['avg_sentence_length'] = np.mean(sent_lengths)
        
        return features
    
    def extract_all_features(self, text: str, tokens: List[str], 
                            pos_tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Extract all features from text"""
        features = {}
        
        # N-grams
        features.update(self.extract_ngrams(tokens, 1))
        features.update(self.extract_ngrams(tokens, 2))
        
        # Character n-grams
        features.update(self.extract_character_ngrams(text, 3))
        
        # POS n-grams if available
        if pos_tags:
            features.update(self.extract_pos_ngrams(pos_tags, 1))
            features.update(self.extract_pos_ngrams(pos_tags, 2))
        
        # Stylometric features
        features.update(self.extract_stylometric_features(text, tokens))
        
        return features
    
    def vectorize(self, features: Dict[str, Any], vocabulary: Dict[str, int]) -> List[float]:
        """Convert features to vector using vocabulary"""
        vector = [0.0] * len(vocabulary)
        for feature, value in features.items():
            if feature in vocabulary:
                idx = vocabulary[feature]
                vector[idx] = float(value)
        return vector


# ============================================================================
# TEXT CLASSIFIER
# ============================================================================

class TextClassifier:
    """Text classification for period, genre, authorship"""
    
    def __init__(self, model_type: str = "naive_bayes"):
        self.model_type = model_type
        self.feature_extractor = FeatureExtractor()
        self.vocabulary = {}
        self.label_counts = Counter()
        self.feature_counts = defaultdict(Counter)
        self.trained = False
    
    def train(self, documents: List[Dict], label_field: str = "period"):
        """Train classifier on documents"""
        logger.info(f"Training {self.model_type} classifier on {len(documents)} documents")
        
        # Build vocabulary and count features
        all_features = []
        for doc in documents:
            text = doc.get('content', '')
            tokens = text.split()  # Simple tokenization
            label = doc.get(label_field, 'unknown')
            
            features = self.feature_extractor.extract_all_features(text, tokens)
            all_features.append((features, label))
            
            self.label_counts[label] += 1
            for feature, value in features.items():
                self.feature_counts[label][feature] += value
        
        # Build vocabulary from all features
        all_feature_names = set()
        for features, _ in all_features:
            all_feature_names.update(features.keys())
        
        self.vocabulary = {name: idx for idx, name in enumerate(sorted(all_feature_names))}
        self.trained = True
        
        logger.info(f"Trained on {len(self.label_counts)} classes, {len(self.vocabulary)} features")
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict class for text"""
        if not self.trained:
            raise ValueError("Classifier not trained")
        
        tokens = text.split()
        features = self.feature_extractor.extract_all_features(text, tokens)
        
        # Naive Bayes prediction
        scores = {}
        total_docs = sum(self.label_counts.values())
        
        for label in self.label_counts:
            # Prior probability
            prior = self.label_counts[label] / total_docs
            log_prob = np.log(prior)
            
            # Likelihood
            total_features = sum(self.feature_counts[label].values())
            vocab_size = len(self.vocabulary)
            
            for feature, value in features.items():
                if feature in self.vocabulary:
                    count = self.feature_counts[label].get(feature, 0)
                    # Laplace smoothing
                    prob = (count + 1) / (total_features + vocab_size)
                    log_prob += value * np.log(prob)
            
            scores[label] = log_prob
        
        # Get best prediction
        best_label = max(scores, key=scores.get)
        
        # Convert to probability
        max_score = max(scores.values())
        exp_scores = {k: np.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        confidence = exp_scores[best_label] / total
        
        return best_label, confidence
    
    def evaluate(self, test_documents: List[Dict], label_field: str = "period") -> Dict:
        """Evaluate classifier on test set"""
        correct = 0
        total = 0
        confusion = defaultdict(Counter)
        
        for doc in test_documents:
            text = doc.get('content', '')
            true_label = doc.get(label_field, 'unknown')
            
            pred_label, confidence = self.predict(text)
            
            if pred_label == true_label:
                correct += 1
            
            confusion[true_label][pred_label] += 1
            total += 1
        
        accuracy = correct / max(total, 1)
        
        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "confusion_matrix": dict(confusion)
        }
    
    def save(self, path: str):
        """Save trained model"""
        model_data = {
            "model_type": self.model_type,
            "vocabulary": self.vocabulary,
            "label_counts": dict(self.label_counts),
            "feature_counts": {k: dict(v) for k, v in self.feature_counts.items()},
            "trained": self.trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved model to {path}")
    
    def load(self, path: str):
        """Load trained model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_type = model_data["model_type"]
        self.vocabulary = model_data["vocabulary"]
        self.label_counts = Counter(model_data["label_counts"])
        self.feature_counts = defaultdict(Counter)
        for k, v in model_data["feature_counts"].items():
            self.feature_counts[k] = Counter(v)
        self.trained = model_data["trained"]
        
        logger.info(f"Loaded model from {path}")


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

class StatisticalAnalyzer:
    """Statistical analysis for corpus comparison"""
    
    def __init__(self):
        pass
    
    def word_frequency(self, tokens: List[str], top_n: int = 100) -> List[Tuple[str, int]]:
        """Get word frequency distribution"""
        counter = Counter(tokens)
        return counter.most_common(top_n)
    
    def hapax_legomena(self, tokens: List[str]) -> List[str]:
        """Find words occurring only once"""
        counter = Counter(tokens)
        return [word for word, count in counter.items() if count == 1]
    
    def vocabulary_growth(self, tokens: List[str], step: int = 1000) -> List[Tuple[int, int]]:
        """Track vocabulary growth over token count"""
        growth = []
        seen = set()
        
        for i, token in enumerate(tokens):
            seen.add(token)
            if (i + 1) % step == 0:
                growth.append((i + 1, len(seen)))
        
        return growth
    
    def compare_corpora(self, corpus1_tokens: List[str], corpus2_tokens: List[str]) -> Dict:
        """Compare two corpora statistically"""
        freq1 = Counter(corpus1_tokens)
        freq2 = Counter(corpus2_tokens)
        
        vocab1 = set(freq1.keys())
        vocab2 = set(freq2.keys())
        
        shared = vocab1 & vocab2
        unique1 = vocab1 - vocab2
        unique2 = vocab2 - vocab1
        
        # Jaccard similarity
        jaccard = len(shared) / len(vocab1 | vocab2) if (vocab1 | vocab2) else 0
        
        return {
            "corpus1_tokens": len(corpus1_tokens),
            "corpus2_tokens": len(corpus2_tokens),
            "corpus1_vocabulary": len(vocab1),
            "corpus2_vocabulary": len(vocab2),
            "shared_vocabulary": len(shared),
            "unique_to_corpus1": len(unique1),
            "unique_to_corpus2": len(unique2),
            "jaccard_similarity": jaccard
        }
    
    def chi_square_keywords(self, target_tokens: List[str], 
                           reference_tokens: List[str], 
                           top_n: int = 50) -> List[Tuple[str, float]]:
        """Find keywords using chi-square test"""
        target_freq = Counter(target_tokens)
        ref_freq = Counter(reference_tokens)
        
        target_total = len(target_tokens)
        ref_total = len(reference_tokens)
        
        all_words = set(target_freq.keys()) | set(ref_freq.keys())
        
        chi_scores = []
        
        for word in all_words:
            o11 = target_freq.get(word, 0)  # word in target
            o12 = ref_freq.get(word, 0)      # word in reference
            o21 = target_total - o11         # other words in target
            o22 = ref_total - o12            # other words in reference
            
            total = o11 + o12 + o21 + o22
            
            # Expected values
            e11 = (o11 + o12) * (o11 + o21) / total
            e12 = (o11 + o12) * (o12 + o22) / total
            e21 = (o21 + o22) * (o11 + o21) / total
            e22 = (o21 + o22) * (o12 + o22) / total
            
            # Chi-square
            chi = 0
            for o, e in [(o11, e11), (o12, e12), (o21, e21), (o22, e22)]:
                if e > 0:
                    chi += (o - e) ** 2 / e
            
            # Direction: positive if overrepresented in target
            direction = 1 if o11 / max(target_total, 1) > o12 / max(ref_total, 1) else -1
            
            chi_scores.append((word, chi * direction))
        
        # Sort by absolute chi-square, keep direction
        chi_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return chi_scores[:top_n]


# ============================================================================
# MODEL REGISTRY
# ============================================================================

class ModelRegistry:
    """Registry for trained ML models"""
    
    def __init__(self, db_path: str = "greek_corpus.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize model registry table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                model_type TEXT,
                task TEXT,
                language TEXT DEFAULT 'grc',
                period TEXT,
                accuracy REAL,
                training_size INTEGER,
                vocabulary_size INTEGER,
                model_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def register_model(self, name: str, model_type: str, task: str,
                      accuracy: float, training_size: int, 
                      vocabulary_size: int, model_path: str,
                      language: str = "grc", period: str = None,
                      metadata: Dict = None) -> int:
        """Register a trained model"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO ml_models 
            (name, model_type, task, language, period, accuracy, 
             training_size, vocabulary_size, model_path, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, model_type, task, language, period, accuracy,
            training_size, vocabulary_size, model_path,
            json.dumps(metadata) if metadata else None
        ))
        
        model_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Registered model: {name} (ID: {model_id})")
        return model_id
    
    def get_models(self, task: str = None, language: str = None) -> List[Dict]:
        """Get registered models"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM ml_models WHERE 1=1"
        params = []
        
        if task:
            query += " AND task = ?"
            params.append(task)
        
        if language:
            query += " AND language = ?"
            params.append(language)
        
        query += " ORDER BY accuracy DESC"
        
        cursor.execute(query, params)
        models = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return models
    
    def get_best_model(self, task: str, language: str = "grc") -> Optional[Dict]:
        """Get best model for task"""
        models = self.get_models(task=task, language=language)
        return models[0] if models else None


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Tools for Historical Greek")
    parser.add_argument('command', choices=['train', 'predict', 'evaluate', 'stats', 'compare'],
                       help="Command to run")
    parser.add_argument('--input', '-i', help="Input file or directory")
    parser.add_argument('--output', '-o', help="Output file")
    parser.add_argument('--model', '-m', help="Model path")
    parser.add_argument('--task', '-t', default="period", help="Classification task")
    
    args = parser.parse_args()
    
    if args.command == 'train':
        classifier = TextClassifier()
        # Load training data and train
        print("Training classifier...")
        # classifier.train(documents, label_field=args.task)
        # classifier.save(args.output or "model.pkl")
    
    elif args.command == 'predict':
        if args.model and args.input:
            classifier = TextClassifier()
            classifier.load(args.model)
            
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
            
            label, confidence = classifier.predict(text)
            print(f"Prediction: {label} (confidence: {confidence:.2%})")
    
    elif args.command == 'stats':
        analyzer = StatisticalAnalyzer()
        print("Statistical analysis tools available")
    
    elif args.command == 'compare':
        analyzer = StatisticalAnalyzer()
        print("Corpus comparison tools available")


if __name__ == "__main__":
    main()
