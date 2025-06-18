"""
Intelligent Recommendation Engine for NoCodeML

This module analyzes datasets and user requirements to recommend optimal
ML algorithms, preprocessing steps, and hyperparameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.models.schemas import (
    ProblemType, DataType, ModelRecommendation, AnalysisResult,
    RecommendationConfig
)
from core.analyzers.dataset_analyzer import DatasetAnalyzer

class RecommendationEngine:
    """
    Intelligent recommendation engine that suggests optimal ML approaches
    based on dataset characteristics and problem requirements.
    """
    
    def __init__(self):
        self.analyzer = DatasetAnalyzer()
        
        # Algorithm performance profiles based on data characteristics
        self.algorithm_profiles = {
            'classification': {
                'RandomForest': {
                    'best_for': ['mixed_features', 'medium_data', 'interpretability'],
                    'data_size_min': 100,
                    'data_size_optimal': 1000,
                    'handles_missing': True,
                    'handles_categorical': True,
                    'confidence_base': 0.85
                },
                'XGBoost': {
                    'best_for': ['structured_data', 'competitions', 'performance'],
                    'data_size_min': 500,
                    'data_size_optimal': 5000,
                    'handles_missing': True,
                    'handles_categorical': True,
                    'confidence_base': 0.90
                },
                'LogisticRegression': {
                    'best_for': ['linear_relationships', 'fast_training', 'interpretability'],
                    'data_size_min': 50,
                    'data_size_optimal': 1000,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'confidence_base': 0.75
                },
                'SVM': {
                    'best_for': ['high_dimensional', 'non_linear', 'small_data'],
                    'data_size_min': 100,
                    'data_size_optimal': 1000,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'confidence_base': 0.80
                },
                'NeuralNetwork': {
                    'best_for': ['complex_patterns', 'large_data', 'non_linear'],
                    'data_size_min': 1000,
                    'data_size_optimal': 10000,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'confidence_base': 0.85
                }
            },
            'regression': {
                'RandomForest': {
                    'best_for': ['mixed_features', 'non_linear', 'robust'],
                    'data_size_min': 100,
                    'data_size_optimal': 1000,
                    'handles_missing': True,
                    'handles_categorical': True,
                    'confidence_base': 0.85
                },
                'XGBoost': {
                    'best_for': ['structured_data', 'competitions', 'performance'],
                    'data_size_min': 500,
                    'data_size_optimal': 5000,
                    'handles_missing': True,
                    'handles_categorical': True,
                    'confidence_base': 0.90
                },
                'LinearRegression': {
                    'best_for': ['linear_relationships', 'interpretability', 'fast'],
                    'data_size_min': 50,
                    'data_size_optimal': 500,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'confidence_base': 0.70
                },
                'Ridge': {
                    'best_for': ['regularization', 'multicollinearity', 'stable'],
                    'data_size_min': 50,
                    'data_size_optimal': 1000,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'confidence_base': 0.75
                }
            }
        }
    
    async def get_recommendations(self, dataset_id: str, config: RecommendationConfig) -> AnalysisResult:
        """
        Generate comprehensive ML recommendations for a dataset
        
        Args:
            dataset_id: ID of the uploaded dataset
            config: User configuration and preferences
            
        Returns:
            AnalysisResult: Complete analysis with model recommendations
        """
        try:
            # Find dataset file
            dataset_files = list((project_root / "data" / "uploads").glob(f"{dataset_id}_*"))
            if not dataset_files:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            dataset_path = str(dataset_files[0])
            
            # Analyze dataset
            dataset_info = await self.analyzer.analyze_dataset(dataset_path)
            df = self.analyzer._load_dataset(dataset_path)
            
            # Get problem-specific recommendations
            recommendations = self._generate_algorithm_recommendations(
                df, dataset_info, config
            )
            
            # Determine preprocessing requirements
            preprocessing_required = self._analyze_preprocessing_needs(df, dataset_info)
            
            # Estimate training time
            estimated_time = self._estimate_training_time(df, recommendations)
            
            return AnalysisResult(
                dataset_id=dataset_id,
                problem_type=config.problem_type,
                target_column=config.target_column,
                feature_columns=config.feature_columns or self._suggest_feature_columns(df, config.target_column),
                recommended_models=recommendations,
                data_preprocessing_required=preprocessing_required,
                estimated_training_time=estimated_time,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            raise Exception(f"Error generating recommendations: {str(e)}")
    
    def _generate_algorithm_recommendations(self, df: pd.DataFrame, dataset_info, 
                                          config: RecommendationConfig) -> List[ModelRecommendation]:
        """Generate algorithm recommendations based on data characteristics"""
        recommendations = []
        
        # Get algorithms for the problem type
        problem_type = config.problem_type.value
        if problem_type not in self.algorithm_profiles:
            return recommendations
        
        algorithms = self.algorithm_profiles[problem_type]
        
        # Analyze data characteristics
        data_characteristics = self._analyze_data_characteristics(df, dataset_info)
        
        # Score each algorithm
        algorithm_scores = {}
        for algorithm_name, profile in algorithms.items():
            score = self._calculate_algorithm_score(data_characteristics, profile)
            algorithm_scores[algorithm_name] = score
        
        # Sort by score and create recommendations
        sorted_algorithms = sorted(algorithm_scores.items(), key=lambda x: x[1], reverse=True)
        
        for algorithm_name, confidence_score in sorted_algorithms[:3]:  # Top 3 recommendations
            profile = algorithms[algorithm_name]
            
            # Generate hyperparameters
            hyperparameters = self._generate_hyperparameters(algorithm_name, data_characteristics)
            
            # Generate preprocessing steps
            preprocessing_steps = self._generate_preprocessing_steps(
                algorithm_name, data_characteristics, profile
            )
            
            # Estimate performance
            expected_performance = self._estimate_performance(
                algorithm_name, data_characteristics, problem_type
            )
            
            # Generate explanation
            explanation = self._generate_explanation(
                algorithm_name, data_characteristics, profile
            )
            
            # Determine algorithm type
            algorithm_type = self._get_algorithm_type(algorithm_name)
            
            recommendation = ModelRecommendation(
                algorithm=algorithm_name,
                algorithm_type=algorithm_type,
                confidence_score=confidence_score,
                expected_performance=expected_performance,
                hyperparameters=hyperparameters,
                preprocessing_steps=preprocessing_steps,
                explanation=explanation
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _analyze_data_characteristics(self, df: pd.DataFrame, dataset_info) -> Dict[str, Any]:
        """Analyze key characteristics of the dataset"""
        characteristics = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'missing_ratio': df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
            'duplicate_ratio': df.duplicated().sum() / len(df),
            'categorical_ratio': len(df.select_dtypes(include=['object']).columns) / len(df.columns),
            'numeric_ratio': len(df.select_dtypes(include=[np.number]).columns) / len(df.columns),
            'data_quality_score': dataset_info.data_quality_score / 100,
            'has_missing_values': df.isnull().any().any(),
            'has_categorical': len(df.select_dtypes(include=['object']).columns) > 0,
            'has_high_cardinality': False,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Check for high cardinality categorical variables
        for col in df.select_dtypes(include=['object']).columns:\n            if df[col].nunique() > len(df) * 0.5:\n                characteristics['has_high_cardinality'] = True\n                break\n        \n        return characteristics\n    \n    def _calculate_algorithm_score(self, data_characteristics: Dict, profile: Dict) -> float:\n        \"\"\"Calculate confidence score for an algorithm based on data characteristics\"\"\"\n        base_score = profile['confidence_base']\n        score = base_score\n        \n        # Data size considerations\n        n_samples = data_characteristics['n_samples']\n        if n_samples < profile['data_size_min']:\n            score *= 0.7  # Penalize if below minimum\n        elif n_samples >= profile['data_size_optimal']:\n            score *= 1.1  # Boost if optimal size\n        \n        # Missing value handling\n        if data_characteristics['has_missing_values'] and not profile['handles_missing']:\n            score *= 0.8\n        \n        # Categorical data handling\n        if data_characteristics['has_categorical'] and not profile['handles_categorical']:\n            score *= 0.85\n        \n        # Data quality considerations\n        score *= data_characteristics['data_quality_score']\n        \n        # High cardinality penalty for some algorithms\n        if data_characteristics['has_high_cardinality'] and profile.get('sensitive_to_cardinality', False):\n            score *= 0.9\n        \n        return min(1.0, max(0.1, score))  # Clamp between 0.1 and 1.0\n    \n    def _generate_hyperparameters(self, algorithm_name: str, data_characteristics: Dict) -> Dict[str, Any]:\n        \"\"\"Generate optimized hyperparameters based on data characteristics\"\"\"\n        n_samples = data_characteristics['n_samples']\n        n_features = data_characteristics['n_features']\n        \n        if algorithm_name == 'RandomForest':\n            n_estimators = min(200, max(50, n_samples // 10))\n            max_depth = min(20, max(5, int(np.log2(n_features)) + 3))\n            return {\n                'n_estimators': n_estimators,\n                'max_depth': max_depth,\n                'min_samples_split': 5 if n_samples > 1000 else 2,\n                'min_samples_leaf': 2 if n_samples > 1000 else 1\n            }\n        \n        elif algorithm_name == 'XGBoost':\n            return {\n                'n_estimators': min(200, max(50, n_samples // 20)),\n                'learning_rate': 0.1 if n_samples > 1000 else 0.2,\n                'max_depth': min(8, max(3, int(np.log2(n_features)))),\n                'subsample': 0.9 if n_samples > 1000 else 1.0\n            }\n        \n        elif algorithm_name == 'LogisticRegression':\n            return {\n                'C': 1.0 if n_samples > 1000 else 10.0,\n                'solver': 'lbfgs' if n_features < 100 else 'liblinear',\n                'max_iter': 1000\n            }\n        \n        elif algorithm_name == 'SVM':\n            return {\n                'C': 1.0,\n                'kernel': 'rbf' if n_features < 100 else 'linear',\n                'gamma': 'scale'\n            }\n        \n        return {}\n    \n    def _generate_preprocessing_steps(self, algorithm_name: str, data_characteristics: Dict, \n                                    profile: Dict) -> List[str]:\n        \"\"\"Generate required preprocessing steps\"\"\"\n        steps = []\n        \n        # Missing value handling\n        if data_characteristics['has_missing_values']:\n            if not profile['handles_missing']:\n                steps.append('Handle Missing Values')\n        \n        # Categorical encoding\n        if data_characteristics['has_categorical']:\n            if not profile['handles_categorical']:\n                if data_characteristics['has_high_cardinality']:\n                    steps.append('Target Encoding')\n                else:\n                    steps.append('One-Hot Encoding')\n        \n        # Scaling for algorithms that need it\n        if algorithm_name in ['LogisticRegression', 'SVM', 'NeuralNetwork']:\n            steps.append('Feature Scaling')\n        \n        # Feature selection for high-dimensional data\n        if data_characteristics['n_features'] > 100:\n            steps.append('Feature Selection')\n        \n        # Outlier detection for sensitive algorithms\n        if algorithm_name in ['LogisticRegression', 'SVM'] and data_characteristics['n_samples'] > 1000:\n            steps.append('Outlier Detection')\n        \n        return steps\n    \n    def _estimate_performance(self, algorithm_name: str, data_characteristics: Dict, \n                            problem_type: str) -> Dict[str, float]:\n        \"\"\"Estimate expected model performance\"\"\"\n        base_performance = {\n            'classification': {\n                'RandomForest': {'accuracy': 0.85, 'f1_score': 0.83},\n                'XGBoost': {'accuracy': 0.87, 'f1_score': 0.85},\n                'LogisticRegression': {'accuracy': 0.78, 'f1_score': 0.76},\n                'SVM': {'accuracy': 0.82, 'f1_score': 0.80}\n            },\n            'regression': {\n                'RandomForest': {'r2_score': 0.82, 'rmse': 0.25},\n                'XGBoost': {'r2_score': 0.85, 'rmse': 0.22},\n                'LinearRegression': {'r2_score': 0.75, 'rmse': 0.30},\n                'Ridge': {'r2_score': 0.77, 'rmse': 0.28}\n            }\n        }\n        \n        if problem_type in base_performance and algorithm_name in base_performance[problem_type]:\n            performance = base_performance[problem_type][algorithm_name].copy()\n            \n            # Adjust based on data quality\n            quality_factor = data_characteristics['data_quality_score']\n            for metric in performance:\n                performance[metric] *= quality_factor\n            \n            return performance\n        \n        return {}\n    \n    def _generate_explanation(self, algorithm_name: str, data_characteristics: Dict, \n                            profile: Dict) -> str:\n        \"\"\"Generate human-readable explanation for algorithm choice\"\"\"\n        explanations = {\n            'RandomForest': f\"Random Forest is recommended because it handles mixed data types well, requires minimal preprocessing, and provides good interpretability. With {data_characteristics['n_samples']} samples, it should perform robustly.\",\n            \n            'XGBoost': f\"XGBoost is highly recommended for structured data with {data_characteristics['n_samples']} samples. It often provides state-of-the-art performance and handles missing values automatically.\",\n            \n            'LogisticRegression': f\"Logistic Regression is suggested for its simplicity and interpretability. It works well when relationships are approximately linear and provides fast training.\",\n            \n            'SVM': f\"Support Vector Machine is recommended for this dataset size ({data_characteristics['n_samples']} samples). It can capture non-linear relationships and works well in high-dimensional spaces.\",\n            \n            'LinearRegression': \"Linear Regression is chosen for its simplicity and interpretability. It works best when the relationship between features and target is approximately linear.\",\n            \n            'Ridge': \"Ridge Regression is recommended to handle potential multicollinearity in your features while maintaining interpretability.\"\n        }\n        \n        base_explanation = explanations.get(algorithm_name, f\"{algorithm_name} is recommended based on your data characteristics.\")\n        \n        # Add data-specific insights\n        if data_characteristics['has_missing_values']:\n            if profile['handles_missing']:\n                base_explanation += \" It naturally handles missing values in your dataset.\"\n            else:\n                base_explanation += \" Note: Missing values will need to be handled during preprocessing.\"\n        \n        if data_characteristics['has_categorical']:\n            if profile['handles_categorical']:\n                base_explanation += \" It can work directly with categorical features.\"\n            else:\n                base_explanation += \" Categorical features will be encoded during preprocessing.\"\n        \n        return base_explanation\n    \n    def _get_algorithm_type(self, algorithm_name: str) -> str:\n        \"\"\"Determine the algorithm category\"\"\"\n        algorithm_types = {\n            'RandomForest': 'ensemble',\n            'XGBoost': 'ensemble',\n            'LogisticRegression': 'classical_ml',\n            'LinearRegression': 'classical_ml',\n            'Ridge': 'classical_ml',\n            'SVM': 'classical_ml',\n            'NeuralNetwork': 'deep_learning'\n        }\n        \n        return algorithm_types.get(algorithm_name, 'classical_ml')\n    \n    def _suggest_feature_columns(self, df: pd.DataFrame, target_column: Optional[str]) -> List[str]:\n        \"\"\"Suggest feature columns if not provided\"\"\"\n        all_columns = df.columns.tolist()\n        \n        if target_column and target_column in all_columns:\n            feature_columns = [col for col in all_columns if col != target_column]\n        else:\n            feature_columns = all_columns\n        \n        # Remove columns that are likely not useful as features\n        excluded_patterns = ['id', 'index', 'key', 'name', 'description']\n        feature_columns = [\n            col for col in feature_columns \n            if not any(pattern in col.lower() for pattern in excluded_patterns)\n        ]\n        \n        return feature_columns\n    \n    def _analyze_preprocessing_needs(self, df: pd.DataFrame, dataset_info) -> bool:\n        \"\"\"Determine if preprocessing is required\"\"\"\n        # Check for missing values\n        if df.isnull().any().any():\n            return True\n        \n        # Check for categorical variables\n        if len(df.select_dtypes(include=['object']).columns) > 0:\n            return True\n        \n        # Check for scale differences in numeric columns\n        numeric_cols = df.select_dtypes(include=[np.number]).columns\n        if len(numeric_cols) > 1:\n            ranges = []\n            for col in numeric_cols:\n                col_range = df[col].max() - df[col].min()\n                if not np.isnan(col_range) and col_range > 0:\n                    ranges.append(col_range)\n            \n            if len(ranges) > 1 and max(ranges) / min(ranges) > 100:\n                return True\n        \n        return False\n    \n    def _estimate_training_time(self, df: pd.DataFrame, recommendations: List[ModelRecommendation]) -> int:\n        \"\"\"Estimate training time in minutes\"\"\"\n        n_samples = len(df)\n        n_features = len(df.columns)\n        \n        # Base time calculation\n        base_time = (n_samples * n_features) / 10000  # minutes\n        \n        # Adjust for algorithm complexity\n        if recommendations:\n            primary_algorithm = recommendations[0].algorithm\n            complexity_multipliers = {\n                'LinearRegression': 0.5,\n                'LogisticRegression': 0.7,\n                'RandomForest': 1.0,\n                'XGBoost': 1.5,\n                'SVM': 2.0,\n                'NeuralNetwork': 3.0\n            }\n            multiplier = complexity_multipliers.get(primary_algorithm, 1.0)\n            base_time *= multiplier\n        \n        # Add preprocessing time\n        base_time += 2  # 2 minutes for preprocessing\n        \n        # Add hyperparameter optimization time\n        base_time += 3  # 3 minutes for hyperparameter tuning\n        \n        return max(1, int(base_time))  # Minimum 1 minute

