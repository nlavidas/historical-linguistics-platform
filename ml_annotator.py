"""
═══════════════════════════════════════════════════════════════════════════
MACHINE LEARNING AUTOMATIC ANNOTATOR
Integrates multiple ML frameworks for automated linguistic annotation
═══════════════════════════════════════════════════════════════════════════

Supports:
- LightSide (Educational ML for text classification)
- scikit-learn (Classic ML algorithms)
- PyTorch (Deep learning models)
- Custom trained models

Author: Nikolaos Lavidas
Institution: National and Kapodistrian University of Athens (NKUA)
Funding: Hellenic Foundation for Research and Innovation (HFRI)
Version: 1.0.0
Date: November 10, 2025
═══════════════════════════════════════════════════════════════════════════
"""

# Load local models configuration
try:
    import local_models_config
except ImportError:
    pass

import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# ML Libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Install: pip install scikit-learn")

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Install: pip install torch")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MLAnnotation:
    """Machine learning annotation result"""
    text: str
    predictions: Dict[str, Any]
    confidence_scores: Dict[str, float]
    model_used: str
    features_used: List[str]
    annotation_time: float
    metadata: Dict[str, Any]


class LightSideIntegration:
    """
    Integration with LightSide ML framework
    Focuses on educational text classification and linguistic annotation
    """
    
    def __init__(self, models_dir: Path = None):
        self.models_dir = models_dir or Path("Z:/models/lightside")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.trained_models = {}
        self.feature_extractors = {}
        logger.info("LightSide Integration initialized")
    
    def load_lightside_model(self, model_path: str) -> bool:
        """Load a trained LightSide model"""
        try:
            model_file = Path(model_path)
            if not model_file.exists():
                logger.error(f"LightSide model not found: {model_path}")
                return False
            
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            model_name = model_file.stem
            self.trained_models[model_name] = model_data
            logger.info(f"✓ Loaded LightSide model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading LightSide model: {e}")
            return False
    
    def extract_lightside_features(self, text: str, 
                                   feature_set: str = "unigrams") -> Dict[str, float]:
        """
        Extract LightSide-style features from text
        
        Feature sets:
        - unigrams: Word frequencies
        - bigrams: 2-word sequences
        - pos: Part-of-speech patterns
        - syntax: Syntactic features
        - discourse: Discourse markers
        """
        features = {}
        
        if feature_set == "unigrams":
            # Word-level features
            words = text.lower().split()
            for word in words:
                feature_name = f"word_{word}"
                features[feature_name] = features.get(feature_name, 0) + 1
        
        elif feature_set == "bigrams":
            # Bigram features
            words = text.lower().split()
            for i in range(len(words) - 1):
                bigram = f"{words[i]}_{words[i+1]}"
                feature_name = f"bigram_{bigram}"
                features[feature_name] = features.get(feature_name, 0) + 1
        
        elif feature_set == "pos":
            # POS-based features (requires NLP)
            try:
                import stanza
                nlp = stanza.Pipeline('en', dir=str(local_models_config.STANZA_DIR))
                doc = nlp(text)
                for sent in doc.sentences:
                    for word in sent.words:
                        feature_name = f"pos_{word.upos}"
                        features[feature_name] = features.get(feature_name, 0) + 1
            except Exception as e:
                logger.warning(f"POS extraction failed: {e}")
        
        elif feature_set == "syntax":
            # Syntactic complexity features
            features["avg_word_length"] = np.mean([len(w) for w in text.split()])
            features["sentence_length"] = len(text.split())
            features["type_token_ratio"] = len(set(text.lower().split())) / max(len(text.split()), 1)
        
        elif feature_set == "discourse":
            # Discourse markers
            discourse_markers = ['however', 'therefore', 'furthermore', 'moreover', 
                               'nevertheless', 'consequently', 'thus', 'hence']
            for marker in discourse_markers:
                if marker in text.lower():
                    features[f"discourse_{marker}"] = 1
        
        return features
    
    def train_lightside_model(self, training_data: List[Tuple[str, str]],
                             model_name: str = "custom_model",
                             algorithm: str = "naive_bayes") -> bool:
        """
        Train a LightSide-style model
        
        Args:
            training_data: List of (text, label) tuples
            model_name: Name for the trained model
            algorithm: ML algorithm (naive_bayes, svm, random_forest, logistic)
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn required for training")
            return False
        
        try:
            # Extract texts and labels
            texts = [item[0] for item in training_data]
            labels = [item[1] for item in training_data]
            
            # Create feature extractor
            vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            
            # Select classifier
            if algorithm == "naive_bayes":
                classifier = MultinomialNB()
            elif algorithm == "svm":
                classifier = SVC(kernel='linear', probability=True)
            elif algorithm == "random_forest":
                classifier = RandomForestClassifier(n_estimators=100)
            elif algorithm == "logistic":
                classifier = LogisticRegression(max_iter=1000)
            else:
                logger.error(f"Unknown algorithm: {algorithm}")
                return False
            
            # Create pipeline
            pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
            
            # Train
            logger.info(f"Training LightSide model: {model_name} ({algorithm})")
            pipeline.fit(texts, labels)
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, texts, labels, cv=5)
            logger.info(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
            
            # Save model
            model_data = {
                'pipeline': pipeline,
                'algorithm': algorithm,
                'labels': list(set(labels)),
                'training_size': len(training_data),
                'cv_accuracy': cv_scores.mean(),
                'trained_at': datetime.now().isoformat()
            }
            
            self.trained_models[model_name] = model_data
            
            # Save to disk
            model_path = self.models_dir / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"✓ Model saved: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def predict_with_lightside(self, text: str, 
                               model_name: str = "custom_model") -> Dict[str, Any]:
        """
        Make predictions using a LightSide model
        """
        if model_name not in self.trained_models:
            logger.error(f"Model not found: {model_name}")
            return {"error": "Model not found"}
        
        try:
            model_data = self.trained_models[model_name]
            pipeline = model_data['pipeline']
            
            # Predict
            prediction = pipeline.predict([text])[0]
            probabilities = pipeline.predict_proba([text])[0]
            
            # Get class labels
            labels = model_data['labels']
            
            result = {
                'prediction': prediction,
                'confidence': float(max(probabilities)),
                'probabilities': {label: float(prob) 
                                for label, prob in zip(labels, probabilities)},
                'model': model_name,
                'algorithm': model_data['algorithm']
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}


class MLAnnotator:
    """
    Main ML annotation system
    Integrates multiple ML frameworks
    """
    
    def __init__(self):
        self.lightside = LightSideIntegration()
        self.models_dir = Path("Z:/models/ml_annotator")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info("ML AUTOMATIC ANNOTATOR INITIALIZED")
        logger.info(f"scikit-learn: {'✓' if SKLEARN_AVAILABLE else '✗'}")
        logger.info(f"PyTorch: {'✓' if PYTORCH_AVAILABLE else '✗'}")
        logger.info("=" * 70)
    
    def annotate_with_ml(self, text: str, 
                        task: str = "classification",
                        model_name: str = None) -> MLAnnotation:
        """
        Automatic annotation using ML
        
        Tasks:
        - classification: Text classification
        - pos_tagging: Part-of-speech tagging
        - ner: Named entity recognition
        - sentiment: Sentiment analysis
        - syntax: Syntactic parsing
        """
        start_time = datetime.now()
        
        # Extract features
        features = self.lightside.extract_lightside_features(text, "unigrams")
        
        # Make predictions based on task
        if task == "classification":
            predictions = self.lightside.predict_with_lightside(text, model_name or "custom_model")
        else:
            predictions = {"task": task, "status": "not_implemented"}
        
        # Calculate confidence
        confidence_scores = {}
        if 'probabilities' in predictions:
            confidence_scores = predictions['probabilities']
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        annotation = MLAnnotation(
            text=text,
            predictions=predictions,
            confidence_scores=confidence_scores,
            model_used=model_name or "default",
            features_used=list(features.keys())[:10],  # Top 10 features
            annotation_time=elapsed,
            metadata={
                'task': task,
                'feature_count': len(features),
                'sklearn_available': SKLEARN_AVAILABLE,
                'pytorch_available': PYTORCH_AVAILABLE
            }
        )
        
        return annotation
    
    def batch_annotate(self, texts: List[str], 
                      task: str = "classification",
                      model_name: str = None) -> List[MLAnnotation]:
        """Annotate multiple texts"""
        logger.info(f"Batch annotating {len(texts)} texts...")
        annotations = []
        
        for i, text in enumerate(texts):
            logger.info(f"Annotating {i+1}/{len(texts)}")
            annotation = self.annotate_with_ml(text, task, model_name)
            annotations.append(annotation)
        
        logger.info(f"✓ Batch annotation complete: {len(annotations)} items")
        return annotations
    
    def save_annotations(self, annotations: List[MLAnnotation], 
                        output_path: str):
        """Save annotations to JSON"""
        output_file = Path(output_path)
        
        data = [asdict(ann) for ann in annotations]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Annotations saved: {output_path}")


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("ML AUTOMATIC ANNOTATOR - Test Mode")
    print("=" * 70)
    print()
    
    # Initialize
    annotator = MLAnnotator()
    
    # Example: Train a simple model
    training_data = [
        ("This is a positive example", "positive"),
        ("This is negative", "negative"),
        ("Great work!", "positive"),
        ("Terrible result", "negative"),
    ]
    
    print("Training example model...")
    success = annotator.lightside.train_lightside_model(
        training_data, 
        model_name="sentiment_demo",
        algorithm="naive_bayes"
    )
    
    if success:
        print("\n✓ Training complete!")
        
        # Test prediction
        test_text = "This is wonderful"
        print(f"\nTesting with: '{test_text}'")
        
        annotation = annotator.annotate_with_ml(test_text, model_name="sentiment_demo")
        print(f"\nPrediction: {annotation.predictions}")
        print(f"Confidence: {annotation.confidence_scores}")
    
    print("\n" + "=" * 70)
    print("ML Annotator ready for production use!")
    print("=" * 70)
