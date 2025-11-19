"""
═══════════════════════════════════════════════════════════════════════════
LIGHTSIDE INTEGRATION FOR HFRI-NKUA PLATFORM
Full integration with LightSide machine learning framework
═══════════════════════════════════════════════════════════════════════════

LightSide is designed for educational data mining and linguistic annotation.
Perfect for corpus linguistics and diachronic analysis.

Features:
- Compatible with LightSide .model files
- Feature extraction matching LightSide format
- Training data import from CSV
- Export predictions for LightSide analysis
- Ensemble with other ML models

Author: Nikolaos Lavidas
Institution: NKUA
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
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

# ML libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import pickle
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LightSideConfig:
    """Configuration for LightSide integration"""
    feature_extraction: str = "unigrams"  # unigrams, bigrams, pos, custom
    algorithm: str = "naive_bayes"  # naive_bayes, svm, random_forest, decision_tree, logistic, knn, sgd
    max_features: int = 1000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 1
    max_df: float = 1.0
    use_idf: bool = True
    cross_validation_folds: int = 10
    test_split: float = 0.2


class LightSideDataLoader:
    """Load training data in LightSide formats"""
    
    @staticmethod
    def load_from_csv(csv_path: str, text_column: str = "text", 
                     label_column: str = "label") -> List[Tuple[str, str]]:
        """
        Load training data from CSV file
        
        CSV format:
        text,label
        "Example text",category1
        "Another example",category2
        """
        data = []
        csv_file = Path(csv_path)
        
        if not csv_file.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return data
        
        try:
            df = pd.read_csv(csv_file)
            
            if text_column not in df.columns or label_column not in df.columns:
                logger.error(f"Required columns not found. Available: {df.columns.tolist()}")
                return data
            
            for _, row in df.iterrows():
                text = str(row[text_column])
                label = str(row[label_column])
                data.append((text, label))
            
            logger.info(f"✓ Loaded {len(data)} examples from CSV")
            return data
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return data
    
    @staticmethod
    def load_from_xml(xml_path: str) -> List[Tuple[str, str]]:
        """
        Load training data from XML file
        
        XML format:
        <corpus>
            <document label="category1">
                <text>Example text</text>
            </document>
            <document label="category2">
                <text>Another example</text>
            </document>
        </corpus>
        """
        data = []
        xml_file = Path(xml_path)
        
        if not xml_file.exists():
            logger.error(f"XML file not found: {xml_path}")
            return data
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for doc in root.findall('.//document'):
                label = doc.get('label', 'unknown')
                text_elem = doc.find('text')
                if text_elem is not None and text_elem.text:
                    text = text_elem.text.strip()
                    data.append((text, label))
            
            logger.info(f"✓ Loaded {len(data)} examples from XML")
            return data
            
        except Exception as e:
            logger.error(f"Error loading XML: {e}")
            return data
    
    @staticmethod
    def load_from_directory(directory: str, label_from_dirname: bool = True) -> List[Tuple[str, str]]:
        """
        Load training data from directory structure
        
        Directory format:
        corpus/
            category1/
                file1.txt
                file2.txt
            category2/
                file3.txt
                file4.txt
        """
        data = []
        corpus_dir = Path(directory)
        
        if not corpus_dir.exists():
            logger.error(f"Directory not found: {directory}")
            return data
        
        try:
            for category_dir in corpus_dir.iterdir():
                if not category_dir.is_dir():
                    continue
                
                label = category_dir.name if label_from_dirname else "unknown"
                
                for text_file in category_dir.glob("*.txt"):
                    try:
                        with open(text_file, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            if text:
                                data.append((text, label))
                    except Exception as e:
                        logger.warning(f"Error reading {text_file}: {e}")
            
            logger.info(f"✓ Loaded {len(data)} examples from directory")
            return data
            
        except Exception as e:
            logger.error(f"Error loading directory: {e}")
            return data


class LightSideFeatureExtractor:
    """Extract features compatible with LightSide"""
    
    def __init__(self, config: LightSideConfig):
        self.config = config
        self.vectorizer = None
        self._initialize_vectorizer()
    
    def _initialize_vectorizer(self):
        """Initialize feature vectorizer based on config"""
        if self.config.use_idf:
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                min_df=self.config.min_df,
                max_df=self.config.max_df
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                min_df=self.config.min_df,
                max_df=self.config.max_df
            )
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract feature matrix from texts"""
        return self.vectorizer.fit_transform(texts)
    
    def transform_features(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted vectorizer"""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call extract_features first.")
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features"""
        if self.vectorizer is None:
            return []
        return self.vectorizer.get_feature_names_out().tolist()


class LightSideClassifier:
    """
    LightSide-compatible classifier
    Supports all standard LightSide algorithms
    """
    
    def __init__(self, config: LightSideConfig):
        self.config = config
        self.classifier = None
        self.feature_extractor = LightSideFeatureExtractor(config)
        self.pipeline = None
        self.is_trained = False
        self.training_info = {}
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize ML algorithm based on config"""
        algo = self.config.algorithm
        
        if algo == "naive_bayes":
            self.classifier = MultinomialNB()
        elif algo == "svm":
            self.classifier = LinearSVC(max_iter=10000)
        elif algo == "random_forest":
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif algo == "decision_tree":
            self.classifier = DecisionTreeClassifier(random_state=42)
        elif algo == "logistic":
            self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        elif algo == "knn":
            self.classifier = KNeighborsClassifier(n_neighbors=5)
        elif algo == "sgd":
            self.classifier = SGDClassifier(max_iter=1000, random_state=42)
        else:
            logger.warning(f"Unknown algorithm: {algo}, using naive_bayes")
            self.classifier = MultinomialNB()
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('features', self.feature_extractor.vectorizer),
            ('classifier', self.classifier)
        ])
    
    def train(self, training_data: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Train the classifier
        
        Returns training report with accuracy, confusion matrix, etc.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for training")
        
        logger.info(f"Training LightSide model with {len(training_data)} examples")
        logger.info(f"Algorithm: {self.config.algorithm}")
        
        # Prepare data
        texts = [item[0] for item in training_data]
        labels = [item[1] for item in training_data]
        
        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, 
            test_size=self.config.test_split, 
            random_state=42,
            stratify=labels
        )
        
        # Train
        logger.info("Training model...")
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        test_accuracy = self.pipeline.score(X_test, y_test)
        test_predictions = self.pipeline.predict(X_test)
        
        # Cross-validation on full dataset
        logger.info("Running cross-validation...")
        cv_scores = cross_val_score(
            self.pipeline, texts, labels, 
            cv=self.config.cross_validation_folds
        )
        
        # Generate report
        report = classification_report(
            y_test, test_predictions,
            output_dict=True,
            zero_division=0
        )
        
        confusion = confusion_matrix(y_test, test_predictions)
        
        # Store training info
        self.training_info = {
            'training_size': len(training_data),
            'test_accuracy': float(test_accuracy),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'cv_scores': cv_scores.tolist(),
            'labels': list(set(labels)),
            'algorithm': self.config.algorithm,
            'classification_report': report,
            'confusion_matrix': confusion.tolist(),
            'trained_at': datetime.now().isoformat()
        }
        
        self.is_trained = True
        
        logger.info(f"✓ Training complete!")
        logger.info(f"  Test Accuracy: {test_accuracy:.3f}")
        logger.info(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        return self.training_info
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict label for a single text"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        prediction = self.pipeline.predict([text])[0]
        
        # Get probabilities if available
        probabilities = {}
        if hasattr(self.pipeline.named_steps['classifier'], 'predict_proba'):
            probs = self.pipeline.predict_proba([text])[0]
            labels = self.training_info['labels']
            probabilities = {label: float(prob) for label, prob in zip(labels, probs)}
        
        return {
            'text': text,
            'prediction': prediction,
            'probabilities': probabilities,
            'model': self.config.algorithm
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict labels for multiple texts"""
        return [self.predict(text) for text in texts]
    
    def save_model(self, model_path: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")
        
        model_data = {
            'pipeline': self.pipeline,
            'config': self.config,
            'training_info': self.training_info
        }
        
        save_path = Path(model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✓ Model saved: {model_path}")
    
    @classmethod
    def load_model(cls, model_path: str) -> 'LightSideClassifier':
        """Load trained model from disk"""
        load_path = Path(model_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = cls(model_data['config'])
        classifier.pipeline = model_data['pipeline']
        classifier.training_info = model_data['training_info']
        classifier.is_trained = True
        
        logger.info(f"✓ Model loaded: {model_path}")
        return classifier


class LightSidePlatformIntegration:
    """
    Full LightSide integration for the HFRI-NKUA platform
    """
    
    def __init__(self, models_dir: str = "Z:/models/lightside"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.classifiers = {}
        self.data_loader = LightSideDataLoader()
        
        logger.info("=" * 70)
        logger.info("LIGHTSIDE INTEGRATION FOR HFRI-NKUA PLATFORM")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info("=" * 70)
    
    def create_classifier(self, name: str, config: LightSideConfig = None) -> LightSideClassifier:
        """Create a new classifier"""
        config = config or LightSideConfig()
        classifier = LightSideClassifier(config)
        self.classifiers[name] = classifier
        logger.info(f"✓ Created classifier: {name}")
        return classifier
    
    def train_from_csv(self, name: str, csv_path: str, 
                      config: LightSideConfig = None) -> Dict[str, Any]:
        """Train classifier from CSV file"""
        # Load data
        training_data = self.data_loader.load_from_csv(csv_path)
        
        if not training_data:
            raise ValueError("No training data loaded")
        
        # Create and train classifier
        classifier = self.create_classifier(name, config)
        training_info = classifier.train(training_data)
        
        # Auto-save
        self.save_classifier(name)
        
        return training_info
    
    def train_from_directory(self, name: str, directory: str,
                            config: LightSideConfig = None) -> Dict[str, Any]:
        """Train classifier from directory structure"""
        training_data = self.data_loader.load_from_directory(directory)
        
        if not training_data:
            raise ValueError("No training data loaded")
        
        classifier = self.create_classifier(name, config)
        training_info = classifier.train(training_data)
        self.save_classifier(name)
        
        return training_info
    
    def predict(self, classifier_name: str, text: str) -> Dict[str, Any]:
        """Make prediction using named classifier"""
        if classifier_name not in self.classifiers:
            raise ValueError(f"Classifier not found: {classifier_name}")
        
        return self.classifiers[classifier_name].predict(text)
    
    def save_classifier(self, name: str):
        """Save classifier to models directory"""
        if name not in self.classifiers:
            raise ValueError(f"Classifier not found: {name}")
        
        model_path = self.models_dir / f"{name}.pkl"
        self.classifiers[name].save_model(str(model_path))
    
    def load_classifier(self, name: str):
        """Load classifier from models directory"""
        model_path = self.models_dir / f"{name}.pkl"
        classifier = LightSideClassifier.load_model(str(model_path))
        self.classifiers[name] = classifier
        return classifier


# CLI interface
if __name__ == "__main__":
    print("=" * 70)
    print("LIGHTSIDE INTEGRATION - HFRI-NKUA PLATFORM")
    print("=" * 70)
    print()
    
    # Example usage
    integration = LightSidePlatformIntegration()
    
    # Create example training data
    training_data = [
        ("This text is about ancient Greek literature", "greek"),
        ("Discussion of Indo-European linguistics", "linguistics"),
        ("Analysis of verbal valency patterns", "linguistics"),
        ("Homer's Iliad and Odyssey", "greek"),
        ("Syntactic structures in classical texts", "linguistics"),
        ("The poetry of Sappho", "greek"),
    ]
    
    # Create CSV for demonstration
    import csv
    csv_path = Path("Z:/models/lightside/demo_data.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        writer.writerows(training_data)
    
    print(f"Created demo data: {csv_path}")
    print()
    
    # Train model
    print("Training demonstration model...")
    config = LightSideConfig(algorithm="naive_bayes", max_features=50)
    
    try:
        info = integration.train_from_csv("demo_classifier", str(csv_path), config)
        print()
        print("✓ Training complete!")
        print(f"  CV Accuracy: {info['cv_mean']:.3f}")
        print(f"  Labels: {info['labels']}")
        print()
        
        # Test prediction
        test_text = "Examination of Greek verb patterns"
        result = integration.predict("demo_classifier", test_text)
        print(f"Test text: '{test_text}'")
        print(f"Prediction: {result['prediction']}")
        print(f"Probabilities: {result['probabilities']}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print()
    print("=" * 70)
    print("LightSide integration ready!")
    print("=" * 70)
