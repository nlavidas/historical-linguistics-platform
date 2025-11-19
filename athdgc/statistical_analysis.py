"""
Statistical Analysis for Diachronic Change Detection
Bootstrap methods and classifier-based approaches
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class DiachronicStatistics:
    """Statistical methods for detecting diachronic linguistic changes"""
    
    def __init__(self):
        self.bootstrap_iterations = 1000
        self.confidence_level = 0.95
    
    def bootstrap_comparison(self, period_a_samples: List[Dict], 
                            period_b_samples: List[Dict],
                            feature: str = 'pattern',
                            iterations: Optional[int] = None) -> Dict:
        """
        Bootstrap method for comparing linguistic features between periods
        
        Args:
            period_a_samples: List of linguistic samples from period A
            period_b_samples: List of linguistic samples from period B
            feature: Feature to analyze (e.g., 'pattern', 'argument_count')
            iterations: Number of bootstrap iterations
        
        Returns:
            Statistical comparison results
        """
        iterations = iterations or self.bootstrap_iterations
        
        # Extract feature values
        values_a = [s.get(feature) for s in period_a_samples if s.get(feature)]
        values_b = [s.get(feature) for s in period_b_samples if s.get(feature)]
        
        if not values_a or not values_b:
            return {'error': 'Insufficient data for comparison'}
        
        # Calculate observed difference
        if isinstance(values_a[0], str):
            # Categorical data - use frequency comparison
            freq_a = Counter(values_a)
            freq_b = Counter(values_b)
            observed_diff = self._calculate_frequency_difference(freq_a, freq_b)
        else:
            # Numerical data - use mean comparison
            observed_diff = np.mean(values_a) - np.mean(values_b)
        
        # Bootstrap resampling
        bootstrap_diffs = []
        combined = values_a + values_b
        n_a = len(values_a)
        
        for _ in range(iterations):
            # Resample
            resample = np.random.choice(combined, size=len(combined), replace=True)
            resample_a = resample[:n_a]
            resample_b = resample[n_a:]
            
            # Calculate difference
            if isinstance(values_a[0], str):
                freq_a_boot = Counter(resample_a)
                freq_b_boot = Counter(resample_b)
                diff = self._calculate_frequency_difference(freq_a_boot, freq_b_boot)
            else:
                diff = np.mean(resample_a) - np.mean(resample_b)
            
            bootstrap_diffs.append(diff)
        
        # Calculate p-value
        bootstrap_diffs = np.array(bootstrap_diffs)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        # Calculate confidence interval
        ci_lower = np.percentile(bootstrap_diffs, (1 - self.confidence_level) * 50)
        ci_upper = np.percentile(bootstrap_diffs, (1 + self.confidence_level) * 50)
        
        result = {
            'feature': feature,
            'observed_difference': float(observed_diff),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'confidence_interval': [float(ci_lower), float(ci_upper)],
            'period_a_n': len(values_a),
            'period_b_n': len(values_b),
            'iterations': iterations
        }
        
        logger.info(f"Bootstrap analysis: {feature} - p={p_value:.4f}, significant={result['significant']}")
        return result
    
    def _calculate_frequency_difference(self, freq_a: Counter, freq_b: Counter) -> float:
        """Calculate frequency difference between two distributions"""
        all_keys = set(freq_a.keys()) | set(freq_b.keys())
        total_diff = 0
        
        for key in all_keys:
            diff = abs(freq_a.get(key, 0) - freq_b.get(key, 0))
            total_diff += diff
        
        return total_diff / len(all_keys) if all_keys else 0
    
    def classifier_method(self, period_a_samples: List[Dict], 
                         period_b_samples: List[Dict],
                         features: List[str]) -> Dict:
        """
        Classifier-based method for detecting diachronic change
        Uses feature importance to identify changing linguistic patterns
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            return {'error': 'scikit-learn not installed. Install with: pip install scikit-learn'}
        
        # Prepare data
        X = []
        y = []
        
        # Extract features
        for sample in period_a_samples:
            feature_vec = self._extract_feature_vector(sample, features)
            if feature_vec:
                X.append(feature_vec)
                y.append(0)  # Period A
        
        for sample in period_b_samples:
            feature_vec = self._extract_feature_vector(sample, features)
            if feature_vec:
                X.append(feature_vec)
                y.append(1)  # Period B
        
        if len(X) < 10:
            return {'error': 'Insufficient samples for classification'}
        
        X = np.array(X)
        y = np.array(y)
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Cross-validation
        scores = cross_val_score(clf, X, y, cv=5)
        
        # Train on full data for feature importance
        clf.fit(X, y)
        
        # Feature importance
        feature_importance = dict(zip(features, clf.feature_importances_))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        result = {
            'accuracy': float(np.mean(scores)),
            'accuracy_std': float(np.std(scores)),
            'feature_importance': {k: float(v) for k, v in sorted_features},
            'top_features': [k for k, v in sorted_features[:5]],
            'n_samples': len(X),
            'significant_change': np.mean(scores) > 0.6  # Above chance
        }
        
        logger.info(f"Classifier method: accuracy={result['accuracy']:.3f}")
        return result
    
    def _extract_feature_vector(self, sample: Dict, features: List[str]) -> Optional[List]:
        """Extract numeric feature vector from sample"""
        vec = []
        
        for feature in features:
            value = sample.get(feature)
            
            if value is None:
                return None
            
            # Convert to numeric
            if isinstance(value, (int, float)):
                vec.append(value)
            elif isinstance(value, str):
                # Hash string to numeric
                vec.append(hash(value) % 1000)
            elif isinstance(value, list):
                vec.append(len(value))
            else:
                vec.append(0)
        
        return vec
    
    def temporal_trend_analysis(self, samples_by_period: Dict[str, List[Dict]],
                               feature: str) -> Dict:
        """
        Analyze trends across multiple time periods
        
        Args:
            samples_by_period: Dict mapping period names to sample lists
            feature: Feature to analyze
        
        Returns:
            Trend analysis results
        """
        periods = sorted(samples_by_period.keys())
        
        if len(periods) < 3:
            return {'error': 'Need at least 3 periods for trend analysis'}
        
        # Calculate feature values per period
        period_values = {}
        
        for period in periods:
            samples = samples_by_period[period]
            values = [s.get(feature) for s in samples if s.get(feature)]
            
            if isinstance(values[0], str):
                # Categorical - use most common
                period_values[period] = Counter(values).most_common(1)[0][0]
            else:
                # Numerical - use mean
                period_values[period] = np.mean(values)
        
        # Detect trend
        values_list = [period_values[p] for p in periods]
        
        if isinstance(values_list[0], str):
            # Categorical trend - count changes
            changes = sum(1 for i in range(len(values_list)-1) if values_list[i] != values_list[i+1])
            trend_type = 'stable' if changes < len(periods) * 0.3 else 'variable'
        else:
            # Numerical trend - calculate slope
            x = np.arange(len(periods))
            y = np.array(values_list)
            slope = np.polyfit(x, y, 1)[0]
            trend_type = 'increasing' if slope > 0.1 else ('decreasing' if slope < -0.1 else 'stable')
        
        result = {
            'feature': feature,
            'periods': periods,
            'values_by_period': period_values,
            'trend_type': trend_type,
            'n_periods': len(periods)
        }
        
        return result
    
    def calculate_effect_size(self, period_a_samples: List[Dict], 
                             period_b_samples: List[Dict],
                             feature: str) -> Dict:
        """Calculate effect size (Cohen's d) for numerical features"""
        values_a = [s.get(feature) for s in period_a_samples if isinstance(s.get(feature), (int, float))]
        values_b = [s.get(feature) for s in period_b_samples if isinstance(s.get(feature), (int, float))]
        
        if not values_a or not values_b:
            return {'error': 'Insufficient numerical data'}
        
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)
        
        # Pooled standard deviation
        std_a = np.std(values_a, ddof=1)
        std_b = np.std(values_b, ddof=1)
        pooled_std = np.sqrt(((len(values_a) - 1) * std_a**2 + (len(values_b) - 1) * std_b**2) / 
                            (len(values_a) + len(values_b) - 2))
        
        # Cohen's d
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            interpretation = 'negligible'
        elif abs(cohens_d) < 0.5:
            interpretation = 'small'
        elif abs(cohens_d) < 0.8:
            interpretation = 'medium'
        else:
            interpretation = 'large'
        
        return {
            'feature': feature,
            'cohens_d': float(cohens_d),
            'interpretation': interpretation,
            'mean_a': float(mean_a),
            'mean_b': float(mean_b),
            'pooled_std': float(pooled_std)
        }
