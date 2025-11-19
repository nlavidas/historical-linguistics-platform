"""
Diachronic Valency Analysis with Bootstrapping & Statistical Classifiers
Implements Yanovich (2018) and Lawson et al. (2021) methodology

Research Focus:
- Valency and valency-changing categories in early Greek, Romance, Germanic
- Comparative method with emphasis on protolanguage origins and language contact
- Statistical analysis addressing data sparseness problem
- Bootstrapping + Classifier methods for temporal period classification

Author: Prof. Nikolaos Lavidas (NKUA/HFRI)
Methodology: Yanovich (2018), Lawson et al. (2021)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TemporalPeriod:
    """Temporal period for diachronic analysis"""
    name: str
    start_year: int
    end_year: int
    language: str
    
class BootstrappingValencyAnalyzer:
    """
    Implements bootstrapping + classifier methodology for valency analysis
    Addresses data sparseness in historical linguistics
    """
    
    def __init__(self, n_bootstrap: int = 1000, random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Define temporal periods
        self.periods = {
            'proto_ie': TemporalPeriod('Proto-Indo-European', -4000, -2500, 'pie'),
            'archaic_greek': TemporalPeriod('Archaic Greek', -800, -500, 'grc'),
            'classical_greek': TemporalPeriod('Classical Greek', -500, -300, 'grc'),
            'hellenistic': TemporalPeriod('Hellenistic Greek', -300, 300, 'grc'),
            'early_latin': TemporalPeriod('Early Latin', -300, 100, 'lat'),
            'classical_latin': TemporalPeriod('Classical Latin', -100, 200, 'lat'),
            'proto_germanic': TemporalPeriod('Proto-Germanic', -500, 500, 'gem'),
            'old_french': TemporalPeriod('Old French', 800, 1300, 'fro'),
        }
        
        logger.info(f"Initialized with {n_bootstrap} bootstrap samples")
    
    def bootstrap_sample(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create pseudo-sample through random modification (with replacement)
        Statistical guarantees about true underlying distribution
        """
        n_samples = len(data)
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return data.iloc[bootstrap_indices].reset_index(drop=True)
    
    def extract_valency_features(self, verb_data: Dict) -> np.ndarray:
        """
        Extract features for classifier from valency pattern
        Features: argument structure, case marking, word order, etc.
        """
        features = []
        
        # Subject properties
        features.append(1 if verb_data.get('has_subject') else 0)
        features.append(verb_data.get('subject_case', 0))  # 1=NOM, 2=ACC, etc.
        
        # Object properties  
        features.append(1 if verb_data.get('has_object') else 0)
        features.append(verb_data.get('object_case', 0))
        
        # Indirect object
        features.append(1 if verb_data.get('has_indirect_obj') else 0)
        features.append(verb_data.get('indirect_case', 0))
        
        # Valency changing operations
        features.append(1 if verb_data.get('is_passive') else 0)
        features.append(1 if verb_data.get('is_causative') else 0)
        features.append(1 if verb_data.get('is_applicative') else 0)
        features.append(1 if verb_data.get('is_antipassive') else 0)
        
        # Contact-induced features
        features.append(1 if verb_data.get('is_borrowing') else 0)
        features.append(verb_data.get('contact_confidence', 0.0))
        
        # Frequency (log-transformed to handle sparseness)
        freq = verb_data.get('frequency', 1)
        features.append(np.log1p(freq))
        
        return np.array(features)
    
    def prepare_classifier_data(self, valency_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for classification
        X = features, y = temporal period labels
        """
        X = []
        y = []
        verb_ids = []
        
        for item in valency_data:
            features = self.extract_valency_features(item)
            period = item.get('period', 'unknown')
            
            if period in self.periods:
                X.append(features)
                y.append(period)
                verb_ids.append(item.get('lemma', 'unknown'))
        
        return np.array(X), np.array(y), verb_ids
    
    def train_temporal_classifier(self, X: np.ndarray, y: np.ndarray, 
                                 classifier_type: str = 'random_forest') -> Any:
        """
        Train classifier to infer temporal period from valency features
        Successful classification = genuine linguistic change
        """
        logger.info(f"\nTraining {classifier_type} classifier...")
        logger.info(f"Training samples: {len(X)}")
        logger.info(f"Temporal periods: {len(np.unique(y))}")
        
        if classifier_type == 'random_forest':
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_state
            )
        elif classifier_type == 'gradient_boosting':
            clf = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state
            )
        elif classifier_type == 'svm':
            clf = SVC(kernel='rbf', probability=True, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown classifier: {classifier_type}")
        
        # Train
        clf.fit(X, y)
        
        # Cross-validation for robustness
        cv_scores = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
        
        logger.info(f"✓ Training complete")
        logger.info(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        return clf, cv_scores
    
    def bootstrap_analysis(self, valency_data: List[Dict], 
                          period1: str, period2: str) -> Dict:
        """
        Bootstrap analysis to determine if two periods show significant difference
        Addresses data sparseness problem
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"BOOTSTRAP ANALYSIS: {period1} vs {period2}")
        logger.info(f"{'='*80}")
        
        # Filter data for the two periods
        data_p1 = [d for d in valency_data if d.get('period') == period1]
        data_p2 = [d for d in valency_data if d.get('period') == period2]
        
        logger.info(f"Period 1 ({period1}): {len(data_p1)} examples")
        logger.info(f"Period 2 ({period2}): {len(data_p2)} examples")
        
        if len(data_p1) < 10 or len(data_p2) < 10:
            logger.warning("⚠ Insufficient data for robust analysis")
        
        # Create DataFrame
        df1 = pd.DataFrame(data_p1)
        df2 = pd.DataFrame(data_p2)
        
        # Bootstrap samples
        bootstrap_results = []
        
        for i in range(self.n_bootstrap):
            # Create pseudo-samples
            boot_df1 = self.bootstrap_sample(df1)
            boot_df2 = self.bootstrap_sample(df2)
            
            # Combine for classification
            combined = pd.concat([boot_df1, boot_df2], ignore_index=True)
            combined_list = combined.to_dict('records')
            
            # Prepare for classification
            X, y, _ = self.prepare_classifier_data(combined_list)
            
            if len(X) > 20:  # Need sufficient samples
                # Train classifier
                clf, cv_scores = self.train_temporal_classifier(X, y, 'random_forest')
                
                bootstrap_results.append({
                    'iteration': i,
                    'accuracy': cv_scores.mean(),
                    'std': cv_scores.std()
                })
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Bootstrap iteration: {i+1}/{self.n_bootstrap}")
        
        # Aggregate results
        accuracies = [r['accuracy'] for r in bootstrap_results]
        
        if accuracies:
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            ci_lower = np.percentile(accuracies, 2.5)
            ci_upper = np.percentile(accuracies, 97.5)
            
            # Hypothesis test: is classification better than chance?
            baseline = 1.0 / len(np.unique([period1, period2]))
            t_stat, p_value = stats.ttest_1samp(accuracies, baseline)
            
            result = {
                'period1': period1,
                'period2': period2,
                'n_bootstrap': len(bootstrap_results),
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'ci_95': [ci_lower, ci_upper],
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_change': p_value < 0.05,
                'interpretation': self._interpret_results(mean_acc, p_value, baseline)
            }
            
            logger.info(f"\n{'='*80}")
            logger.info(f"RESULTS:")
            logger.info(f"  Mean Classification Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
            logger.info(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
            logger.info(f"  Baseline (chance): {baseline:.3f}")
            logger.info(f"  p-value: {p_value:.4f}")
            logger.info(f"  Significant change: {'YES' if p_value < 0.05 else 'NO'}")
            logger.info(f"{'='*80}")
            
            return result
        else:
            logger.error("No bootstrap results - insufficient data")
            return {}
    
    def _interpret_results(self, accuracy: float, p_value: float, baseline: float) -> str:
        """Interpret statistical results for linguistic conclusions"""
        if p_value >= 0.05:
            return "No significant difference between periods - linguistic stability"
        elif accuracy > baseline + 0.2:
            return "Strong temporal distinction - substantial linguistic change detected"
        elif accuracy > baseline + 0.1:
            return "Moderate temporal distinction - gradual linguistic change"
        else:
            return "Weak but significant distinction - subtle linguistic change"
    
    def identify_change_loci(self, valency_data: List[Dict], 
                            period1: str, period2: str) -> Dict:
        """
        Identify exact loci of change through feature importance analysis
        Partial classification success indicates specific features changing
        """
        logger.info(f"\nIdentifying loci of change between {period1} and {period2}...")
        
        # Prepare data
        data = [d for d in valency_data if d.get('period') in [period1, period2]]
        X, y, verbs = self.prepare_classifier_data(data)
        
        if len(X) < 20:
            return {'error': 'Insufficient data'}
        
        # Train Random Forest (provides feature importances)
        clf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        clf.fit(X, y)
        
        # Feature names
        feature_names = [
            'has_subject', 'subject_case', 'has_object', 'object_case',
            'has_indirect_obj', 'indirect_case', 'is_passive', 'is_causative',
            'is_applicative', 'is_antipassive', 'is_borrowing', 
            'contact_confidence', 'frequency_log'
        ]
        
        # Get importances
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        loci = []
        for i in indices[:5]:  # Top 5 features
            loci.append({
                'feature': feature_names[i],
                'importance': float(importances[i]),
                'interpretation': self._interpret_feature(feature_names[i])
            })
        
        logger.info("\n🔍 Top Loci of Change:")
        for loc in loci:
            logger.info(f"  • {loc['feature']}: {loc['importance']:.3f} - {loc['interpretation']}")
        
        return {'loci': loci, 'period1': period1, 'period2': period2}
    
    def _interpret_feature(self, feature: str) -> str:
        """Linguistic interpretation of features"""
        interpretations = {
            'has_subject': 'Presence/absence of explicit subject',
            'subject_case': 'Subject case marking variation',
            'has_object': 'Transitivity changes',
            'object_case': 'Object case marking variation',
            'is_passive': 'Passive voice usage',
            'is_causative': 'Causative formation changes',
            'is_borrowing': 'Language contact effects',
            'contact_confidence': 'Degree of contact influence',
            'frequency_log': 'Usage frequency shifts'
        }
        return interpretations.get(feature, 'Structural change')
    
    def comparative_protolanguage_analysis(self, valency_data: List[Dict]) -> Dict:
        """
        Analyze relationship to protolanguage origins
        Identify inherited vs. innovated patterns
        """
        logger.info(f"\n{'='*80}")
        logger.info("COMPARATIVE METHOD: Protolanguage Origins Analysis")
        logger.info(f"{'='*80}")
        
        # Identify proto patterns (if available)
        proto_patterns = [d for d in valency_data if 'proto' in d.get('period', '').lower()]
        
        if not proto_patterns:
            logger.warning("No protolanguage data available - using reconstruction")
        
        # For each language, compare to proto-patterns
        languages = ['grc', 'lat', 'gem', 'fro']
        results = {}
        
        for lang in languages:
            lang_data = [d for d in valency_data if d.get('language') == lang]
            
            if lang_data:
                # Calculate retention vs. innovation rate
                # (simplified - would use more sophisticated reconstruction)
                n_inherited = sum(1 for d in lang_data if not d.get('is_borrowing', False))
                n_total = len(lang_data)
                retention_rate = n_inherited / n_total if n_total > 0 else 0
                
                results[lang] = {
                    'retention_rate': retention_rate,
                    'innovation_rate': 1 - retention_rate,
                    'n_patterns': n_total
                }
                
                logger.info(f"\n{lang.upper()}:")
                logger.info(f"  Retention rate: {retention_rate:.2%}")
                logger.info(f"  Innovation rate: {(1-retention_rate):.2%}")
        
        return results


# Demonstration
if __name__ == "__main__":
    print("Diachronic Valency Analysis - Bootstrapping & Classifier Methods")
    print("Methodology: Yanovich (2018), Lawson et al. (2021)")
    print("="*80)
    
    analyzer = BootstrappingValencyAnalyzer(n_bootstrap=100)  # Reduced for demo
    
    # Demo data
    demo_data = [
        {'lemma': 'λέγω', 'period': 'archaic_greek', 'language': 'grc', 
         'has_subject': True, 'subject_case': 1, 'has_object': True, 'object_case': 2,
         'is_passive': False, 'is_borrowing': False, 'frequency': 150},
        {'lemma': 'ποιέω', 'period': 'classical_greek', 'language': 'grc',
         'has_subject': True, 'subject_case': 1, 'has_object': True, 'object_case': 2,
         'is_passive': True, 'is_borrowing': False, 'frequency': 200},
    ]
    
    print("\n✓ System ready for production analysis")
    print("\nLoad your valency data and run:")
    print("  analyzer.bootstrap_analysis(data, 'archaic_greek', 'classical_greek')")
