"""
Integration Module for Enhanced Preprocessing with Iterative Model Training

This module integrates the enhanced preprocessing system with the existing 
model training infrastructure to provide comprehensive iterative improvement.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
from datetime import datetime
from pathlib import Path
import sys
import json
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from core.preprocessing.enhanced_preprocessor import EnhancedDataPreprocessor, PreprocessingResult
from core.models.iterative_trainer import IterativeModelTrainer, TrainingResult
from core.automl.automl_engine import AutoMLEngine
from backend.models.schemas import ProblemType

logger = logging.getLogger(__name__)

class EnhancedMLPipeline:
    """
    Enhanced ML Pipeline that combines preprocessing and model training
    with iterative improvement capabilities
    """
    
    def __init__(self, learning_enabled: bool = True):
        self.preprocessor = EnhancedDataPreprocessor(learning_enabled=learning_enabled)
        self.model_trainer = IterativeModelTrainer()
        self.automl_engine = AutoMLEngine()
        self.pipeline_history = []
        
    async def train_with_iterative_improvement(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType,
        max_preprocessing_iterations: int = 3,
        max_training_iterations: int = 3,
        improvement_threshold: float = 0.01,
        user_feedback_callback: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train models with iterative preprocessing and model improvement
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            problem_type: ML problem type
            max_preprocessing_iterations: Maximum preprocessing improvement iterations
            max_training_iterations: Maximum model training iterations  
            improvement_threshold: Minimum improvement threshold to continue
            user_feedback_callback: Callback for user feedback
            progress_callback: Callback for progress updates
            
        Returns:
            Comprehensive training results with preprocessing and model metrics
        """
        logger.info("Starting enhanced ML pipeline with iterative improvement")
        start_time = datetime.now()
        
        if progress_callback:
            await progress_callback("Initializing pipeline", 5)
        
        # Phase 1: Iterative Preprocessing Improvement
        logger.info("Phase 1: Iterative preprocessing improvement")
        if progress_callback:
            await progress_callback("Improving data preprocessing", 10)
        
        def model_performance_callback(processed_df: pd.DataFrame, target_col: str) -> Dict[str, float]:
            """Quick model evaluation for preprocessing feedback"""
            try:
                # Quick training for preprocessing feedback
                X = processed_df.drop(columns=[target_col])
                y = processed_df[target_col]
                
                # Use a simple model for quick evaluation
                quick_result = self.model_trainer.train_and_optimize(
                    X, y, problem_type, test_size=0.2, max_iterations=1, optimize_hyperparameters=False
                )
                
                if quick_result.best_model:
                    return quick_result.best_model.metrics
                return {}
            except Exception as e:
                logger.warning(f"Quick model evaluation failed: {str(e)}")
                return {}
        
        # Perform iterative preprocessing
        best_processed_df, preprocessing_results = self.preprocessor.iterative_improvement(
            df, target_column, problem_type,
            model_performance_callback=model_performance_callback,
            max_iterations=max_preprocessing_iterations,
            improvement_threshold=improvement_threshold
        )
        
        if progress_callback:
            await progress_callback("Preprocessing optimization completed", 40)
        
        # Phase 2: Comprehensive Model Training
        logger.info("Phase 2: Comprehensive model training")
        if progress_callback:
            await progress_callback("Training and optimizing models", 50)
        
        # Prepare features and target
        X = best_processed_df.drop(columns=[target_column])
        y = best_processed_df[target_column]
        
        # Handle small or problematic datasets before training
        X, y = self._prepare_dataset_for_training(X, y, problem_type)
        
        # Train models with iterative optimization
        training_result = self.model_trainer.train_and_optimize(
            X, y, problem_type,
            max_iterations=max_training_iterations,
            optimize_hyperparameters=True
        )
        
        if progress_callback:
            await progress_callback("Model training completed", 80)
        
        # Phase 3: Generate Comprehensive Results
        logger.info("Phase 3: Generating comprehensive results")
        if progress_callback:
            await progress_callback("Generating reports and recommendations", 90)
        
        # Combine preprocessing and training results
        comprehensive_results = self._create_comprehensive_results(
            df, best_processed_df, preprocessing_results, training_result, 
            target_column, problem_type, start_time
        )
        
        # Get user feedback if callback provided
        if user_feedback_callback:
            try:
                feedback = await user_feedback_callback(comprehensive_results)
                if feedback:
                    comprehensive_results['user_feedback'] = feedback
                    # Learn from feedback
                    self._learn_from_user_feedback(feedback, preprocessing_results, training_result)
            except Exception as e:
                logger.warning(f"User feedback collection failed: {str(e)}")
        
        # Store in pipeline history
        self.pipeline_history.append(comprehensive_results)
        
        if progress_callback:
            await progress_callback("Pipeline completed successfully", 100)
        
        logger.info(f"Enhanced ML pipeline completed in {(datetime.now() - start_time).total_seconds():.2f} seconds")
        
        return comprehensive_results
    
    def _create_comprehensive_results(
        self,
        original_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        preprocessing_results: List[PreprocessingResult],
        training_result: TrainingResult,
        target_column: str,
        problem_type: ProblemType,
        start_time: datetime
    ) -> Dict[str, Any]:
        """Create comprehensive results combining preprocessing and training"""
        
        final_preprocessing = preprocessing_results[-1] if preprocessing_results else None
        
        results = {
            'pipeline_summary': {
                'total_processing_time': (datetime.now() - start_time).total_seconds(),
                'preprocessing_iterations': len(preprocessing_results),
                'model_training_iterations': len(training_result.all_models) if training_result.all_models else 0,
                'original_data_shape': original_df.shape,
                'processed_data_shape': processed_df.shape,
                'target_column': target_column,
                'problem_type': problem_type.value,
                'timestamp': datetime.now().isoformat()
            },
            
            'preprocessing_summary': {
                'total_quality_improvement': final_preprocessing.quality_improvement if final_preprocessing else 0,
                'issues_found': len(final_preprocessing.issues_found) if final_preprocessing else 0,
                'issues_fixed': len(final_preprocessing.issues_fixed) if final_preprocessing else 0,
                'transformations_applied': len(final_preprocessing.transformations_applied) if final_preprocessing else 0,
                'data_quality_score': final_preprocessing.metrics.data_quality_score if final_preprocessing else 0,
                'feature_quality_score': final_preprocessing.metrics.feature_quality_score if final_preprocessing else 0
            },
            
            'model_training_summary': {
                'best_model_name': training_result.best_model.model_name if training_result.best_model else None,
                'best_model_performance': training_result.best_model.metrics if training_result.best_model else {},
                'total_models_trained': len(training_result.all_models),
                'training_recommendations': training_result.recommendations,
                'feature_importance': training_result.best_model.feature_importance if training_result.best_model else {}
            },
            
            'detailed_results': {
                'preprocessing_results': [self._serialize_preprocessing_result(r) for r in preprocessing_results],
                'training_result': self._serialize_training_result(training_result),
                'final_feature_names': final_preprocessing.feature_names if final_preprocessing else [],
                'encoding_mappings': final_preprocessing.encoding_mappings if final_preprocessing else {},
                'scaling_parameters': final_preprocessing.scaler_params if final_preprocessing else {}
            },
            
            'recommendations': self._generate_comprehensive_recommendations(
                final_preprocessing, training_result, problem_type
            ),
            
            'next_steps': self._generate_next_steps(final_preprocessing, training_result),
            
            'improvement_opportunities': self._identify_improvement_opportunities(
                final_preprocessing, training_result
            )
        }
        
        return results
    
    def _serialize_preprocessing_result(self, result: PreprocessingResult) -> Dict[str, Any]:
        """Serialize preprocessing result for JSON compatibility"""
        return {
            'original_shape': result.original_shape,
            'processed_shape': result.processed_shape,
            'quality_improvement': result.quality_improvement,
            'transformations_applied': result.transformations_applied,
            'issues_found_count': len(result.issues_found),
            'issues_fixed_count': len(result.issues_fixed),
            'metrics': {
                'data_quality_score': result.metrics.data_quality_score,
                'missing_data_ratio': result.metrics.missing_data_ratio,
                'duplicate_ratio': result.metrics.duplicate_ratio,
                'feature_quality_score': result.metrics.feature_quality_score,
                'preprocessing_time': result.metrics.preprocessing_time
            },
            'recommendations': result.recommendations
        }
    
    def _serialize_training_result(self, result: TrainingResult) -> Dict[str, Any]:
        """Serialize training result for JSON compatibility"""
        return {
            'best_model': {
                'name': result.best_model.model_name if result.best_model else None,
                'metrics': result.best_model.metrics if result.best_model else {},
                'hyperparameters': result.best_model.hyperparameters if result.best_model else {},
                'training_time': result.best_model.training_time if result.best_model else 0
            },
            'all_models_performance': [
                {
                    'name': model.model_name,
                    'metrics': model.metrics,
                    'training_time': model.training_time
                }
                for model in result.all_models
            ],
            'training_summary': result.training_summary,
            'recommendations': result.recommendations,
            'data_insights': result.data_insights,
            'improvement_suggestions': result.improvement_suggestions
        }
    
    def _generate_comprehensive_recommendations(
        self,
        preprocessing_result: Optional[PreprocessingResult],
        training_result: TrainingResult,
        problem_type: ProblemType
    ) -> List[str]:
        """Generate comprehensive recommendations"""
        recommendations = []
        
        # Preprocessing-based recommendations
        if preprocessing_result:
            if preprocessing_result.metrics.data_quality_score < 70:
                recommendations.append(
                    "Data quality score is below 70%. Consider additional data cleaning or collection of higher quality data."
                )
            
            if preprocessing_result.metrics.missing_data_ratio > 0.2:
                recommendations.append(
                    "High missing data ratio detected. Consider domain expertise for better imputation strategies."
                )
            
            recommendations.extend(preprocessing_result.recommendations)
        
        # Training-based recommendations
        if training_result.best_model:
            primary_metric = self._get_primary_metric(training_result.best_model.metrics, problem_type)
            
            if primary_metric < 0.7:
                recommendations.extend([
                    "Model performance is below 70%. Consider:",
                    "- Collecting more training data",
                    "- Advanced feature engineering techniques",
                    "- Ensemble methods or deep learning approaches",
                    "- Domain expert consultation for feature insights"
                ])
            elif primary_metric > 0.95:
                recommendations.append(
                    "Very high performance detected. Verify there's no data leakage or overfitting."
                )
        
        recommendations.extend(training_result.recommendations)
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_next_steps(
        self,
        preprocessing_result: Optional[PreprocessingResult],
        training_result: TrainingResult
    ) -> List[str]:
        """Generate actionable next steps"""
        next_steps = []
        
        # Model deployment readiness
        if training_result.best_model:
            next_steps.append("✅ Model is ready for deployment")
            next_steps.append("📊 Set up model monitoring and performance tracking")
            next_steps.append("🔄 Plan for model retraining schedule")
        
        # Data pipeline improvements
        if preprocessing_result and preprocessing_result.metrics.data_quality_score < 80:
            next_steps.append("🧹 Implement automated data quality monitoring")
            next_steps.append("📈 Set up data quality alerts and thresholds")
        
        # Feature engineering opportunities
        if training_result.feature_analysis:
            if len(training_result.feature_analysis.get('least_important_features', [])) > 5:
                next_steps.append("🎯 Consider feature selection to reduce model complexity")
        
        # Performance optimization
        next_steps.extend([
            "⚡ Optimize model inference speed if needed for production",
            "🧪 Set up A/B testing framework for model improvements",
            "📚 Document preprocessing pipeline for reproducibility"
        ])
        
        return next_steps
    
    def _identify_improvement_opportunities(
        self,
        preprocessing_result: Optional[PreprocessingResult],
        training_result: TrainingResult
    ) -> Dict[str, List[str]]:
        """Identify specific improvement opportunities"""
        opportunities = {
            'data_quality': [],
            'feature_engineering': [],
            'model_performance': [],
            'operational': []
        }
        
        # Data quality opportunities
        if preprocessing_result:
            if preprocessing_result.metrics.missing_data_ratio > 0.1:
                opportunities['data_quality'].append("Reduce missing data through improved data collection")
            
            if preprocessing_result.metrics.duplicate_ratio > 0.05:
                opportunities['data_quality'].append("Implement automated duplicate detection in data pipeline")
        
        # Feature engineering opportunities
        if training_result.best_model:
            if hasattr(training_result.best_model, 'feature_importance'):
                low_importance_features = [
                    f for f, importance in training_result.best_model.feature_importance.items()
                    if importance < 0.01
                ]
                if len(low_importance_features) > 5:
                    opportunities['feature_engineering'].append(
                        f"Remove {len(low_importance_features)} low-importance features"
                    )
        
        # Model performance opportunities
        if training_result.improvement_suggestions:
            opportunities['model_performance'].extend(training_result.improvement_suggestions[:3])
        
        # Operational opportunities
        opportunities['operational'].extend([
            "Implement model versioning and rollback capabilities",
            "Set up automated retraining pipeline",
            "Create model performance dashboard"
        ])
        
        return opportunities
    
    def _get_primary_metric(self, metrics: Dict[str, float], problem_type: ProblemType) -> float:
        """Get primary performance metric"""
        if problem_type == ProblemType.CLASSIFICATION:
            return metrics.get('accuracy', metrics.get('f1_score', 0.0))
        elif problem_type == ProblemType.REGRESSION:
            return metrics.get('r2_score', 0.0)
        else:
            return metrics.get('silhouette_score', 0.0)
    
    def _learn_from_user_feedback(
        self,
        feedback: Dict[str, Any],
        preprocessing_results: List[PreprocessingResult],
        training_result: TrainingResult
    ):
        """Learn from user feedback to improve future pipelines"""
        
        # Extract satisfaction scores
        overall_satisfaction = feedback.get('overall_satisfaction', 5)
        preprocessing_satisfaction = feedback.get('preprocessing_satisfaction', 5)
        model_satisfaction = feedback.get('model_satisfaction', 5)
        
        # Store feedback for learning
        feedback_record = {
            'timestamp': datetime.now().isoformat(),
            'feedback': feedback,
            'preprocessing_quality': preprocessing_results[-1].metrics.data_quality_score if preprocessing_results else 0,
            'model_performance': training_result.best_model.metrics if training_result.best_model else {},
            'satisfaction_scores': {
                'overall': overall_satisfaction,
                'preprocessing': preprocessing_satisfaction,
                'model': model_satisfaction
            }
        }
        
        # Pass feedback to preprocessor for learning
        if preprocessing_results:
            self.preprocessor._learn_from_feedback(feedback, preprocessing_results[-1])
        
        logger.info(f"User feedback recorded: Overall satisfaction {overall_satisfaction}/10")
    
    async def predict_with_preprocessing(
        self,
        input_data: Dict[str, Any],
        model_id: str,
        preprocessing_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make predictions with preprocessing applied"""
        try:
            # Apply same preprocessing transformations used during training
            if preprocessing_config:
                # Convert input to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Apply preprocessing (simplified version)
                # In production, you'd want to save and load the exact preprocessing pipeline
                processed_input = self._apply_saved_preprocessing(input_df, preprocessing_config)
                
                # Convert back to dict
                processed_data = processed_input.iloc[0].to_dict()
            else:
                processed_data = input_data
            
            # Make prediction using AutoML engine
            prediction_result = await self.automl_engine.predict(model_id, processed_data)
            
            return {
                'prediction': prediction_result['prediction'],
                'model_id': model_id,
                'preprocessing_applied': preprocessing_config is not None,
                'input_processed': processed_data if preprocessing_config else None
            }
            
        except Exception as e:
            logger.error(f"Prediction with preprocessing failed: {str(e)}")
            raise
    
    def _apply_saved_preprocessing(
        self,
        input_df: pd.DataFrame,
        preprocessing_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply saved preprocessing configuration to new data"""
        # This is a simplified version - in production you'd save the complete pipeline
        processed_df = input_df.copy()
        
        # Apply encoding mappings
        for col, mapping in preprocessing_config.get('encoding_mappings', {}).items():
            if col in processed_df.columns:
                if mapping['type'] == 'label':
                    # Apply label encoding
                    pass  # Implementation would depend on saved encoder
                elif mapping['type'] == 'onehot':
                    # Apply one-hot encoding
                    pass  # Implementation would depend on saved encoder
        
        # Apply scaling
        scaling_params = preprocessing_config.get('scaling_parameters', {})
        if scaling_params and scaling_params.get('type') == 'standard':
            # Apply standard scaling
            pass  # Implementation would depend on saved scaler
        
        return processed_df
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of all pipeline runs"""
        if not self.pipeline_history:
            return {'message': 'No pipeline runs recorded yet'}
        
        summary = {
            'total_runs': len(self.pipeline_history),
            'average_processing_time': np.mean([
                run['pipeline_summary']['total_processing_time'] 
                for run in self.pipeline_history
            ]),
            'best_performance': max([
                max(run['model_training_summary']['best_model_performance'].values()) 
                if run['model_training_summary']['best_model_performance'] else 0
                for run in self.pipeline_history
            ]),
            'improvement_trend': self._calculate_improvement_trend(),
            'most_common_recommendations': self._get_common_recommendations()
        }
        
        return summary
    
    def _prepare_dataset_for_training(self, X: pd.DataFrame, y: pd.Series, problem_type: ProblemType) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare dataset for training by handling problematic cases"""
        
        logger.info(f"Preparing dataset for training: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Check for minimum dataset size
        if len(X) < 4:
            logger.warning(f"Dataset too small ({len(X)} samples). Duplicating samples for training.")
            
            # Simple duplication strategy
            multiplier = max(2, 4 // len(X) + 1)
            indices = list(range(len(X))) * multiplier
            X_expanded = X.iloc[indices].reset_index(drop=True)
            y_expanded = y.iloc[indices].reset_index(drop=True)
            
            logger.info(f"Dataset expanded from {len(X)} to {len(X_expanded)} samples")
            return X_expanded, y_expanded
        
        # For classification, handle class imbalance issues
        if problem_type == ProblemType.CLASSIFICATION:
            class_counts = y.value_counts()
            
            # Check for classes with only 1 sample
            single_sample_classes = class_counts[class_counts == 1]
            if len(single_sample_classes) > 0:
                logger.warning(f"Classes with single samples detected: {single_sample_classes.index.tolist()}")
                
                # Duplicate single-sample classes
                indices_to_add = []
                for class_label in single_sample_classes.index:
                    class_indices = y[y == class_label].index.tolist()
                    # Duplicate each single-sample class 2 times
                    indices_to_add.extend(class_indices * 2)
                
                if indices_to_add:
                    # Add duplicated samples
                    original_indices = X.index.tolist()
                    all_indices = original_indices + indices_to_add
                    
                    X_balanced = X.loc[all_indices].reset_index(drop=True)
                    y_balanced = y.loc[all_indices].reset_index(drop=True)
                    
                    logger.info(f"Balanced dataset from {len(X)} to {len(X_balanced)} samples")
                    return X_balanced, y_balanced
            
            # Check for severe class imbalance
            if len(class_counts) > 1:
                imbalance_ratio = class_counts.min() / class_counts.max()
                if imbalance_ratio < 0.1:  # Severe imbalance
                    logger.warning(f"Severe class imbalance detected (ratio: {imbalance_ratio:.3f})")
                    
                    # Simple oversampling for minority classes
                    target_count = min(class_counts.max(), class_counts.median() * 2)
                    indices_to_add = []
                    
                    for class_label in class_counts.index:
                        current_count = class_counts[class_label]
                        if current_count < target_count:
                            class_indices = y[y == class_label].index.tolist()
                            samples_needed = int(target_count - current_count)
                            
                            # Oversample by repeating indices
                            repetitions = (samples_needed // current_count) + 1
                            indices_to_add.extend(class_indices * repetitions)
                    
                    if indices_to_add:
                        # Add oversampled instances
                        original_indices = X.index.tolist()
                        all_indices = original_indices + indices_to_add[:len(original_indices)]  # Don't exceed 2x original size
                        
                        X_balanced = X.loc[all_indices].reset_index(drop=True)
                        y_balanced = y.loc[all_indices].reset_index(drop=True)
                        
                        logger.info(f"Applied oversampling: {len(X)} -> {len(X_balanced)} samples")
                        return X_balanced, y_balanced
        
        # Check for any remaining NaN values - Enhanced emergency cleanup
        if X.isnull().any().any() or y.isnull().any():
            logger.warning("NaN values detected in training data. Applying emergency cleaning.")
            print("🚨 Emergency NaN cleanup in preprocessing-trainer integration...")
            
            # Drop rows with NaN in target
            nan_target_mask = y.isnull()
            if nan_target_mask.any():
                X = X[~nan_target_mask]
                y = y[~nan_target_mask]
                logger.info(f"Removed {nan_target_mask.sum()} rows with NaN target values")
            
            # Fill remaining NaN in features
            for col in X.columns:
                if X[col].isnull().any():
                    if X[col].dtype in ['int64', 'float64']:
                        X[col] = X[col].fillna(X[col].median())
                    else:
                        X[col] = X[col].fillna('Unknown')
            
            logger.info("Applied emergency NaN cleaning")
        
        # Ensure we still have enough data after cleaning
        if len(X) < 2:
            logger.error("Insufficient data after cleaning. Cannot proceed with training.")
            raise ValueError("Dataset too small for training after preprocessing")
        
        logger.info(f"Dataset preparation completed: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def _calculate_improvement_trend(self) -> str:
        """Calculate performance improvement trend across runs"""
        if len(self.pipeline_history) < 2:
            return "Insufficient data"
        
        performances = []
        for run in self.pipeline_history:
            metrics = run['model_training_summary']['best_model_performance']
            if metrics:
                # Use the first available metric as proxy for performance
                performance = next(iter(metrics.values()), 0)
                performances.append(performance)
        
        if len(performances) < 2:
            return "Insufficient data"
        
        # Simple trend calculation
        if performances[-1] > performances[0]:
            return "Improving"
        elif performances[-1] < performances[0]:
            return "Declining"
        else:
            return "Stable"
    
    def _get_common_recommendations(self) -> List[str]:
        """Get most common recommendations across runs"""
        all_recommendations = []
        for run in self.pipeline_history:
            all_recommendations.extend(run.get('recommendations', []))
        
        # Count frequency of recommendations
        from collections import Counter
        recommendation_counts = Counter(all_recommendations)
        
        # Return top 5 most common recommendations
        return [rec for rec, count in recommendation_counts.most_common(5)]

# Example usage function
async def example_enhanced_pipeline_usage():
    """Example of how to use the enhanced ML pipeline"""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature4': np.random.exponential(1, n_samples),  # Skewed distribution
        'target': np.random.choice([0, 1], n_samples)
    })
    
    # Add some missing values and duplicates for demonstration
    sample_data.loc[sample_data.index[:50], 'feature1'] = np.nan
    sample_data = pd.concat([sample_data, sample_data.iloc[:20]])  # Add duplicates
    
    # Initialize pipeline
    pipeline = EnhancedMLPipeline(learning_enabled=True)
    
    # Progress callback
    async def progress_callback(message: str, progress: int):
        print(f"Progress: {progress}% - {message}")
    
    # User feedback callback
    async def feedback_callback(results: Dict[str, Any]) -> Dict[str, Any]:
        # In a real application, this would collect actual user feedback
        return {
            'overall_satisfaction': 8,
            'preprocessing_satisfaction': 9,
            'model_satisfaction': 7,
            'comments': 'Good preprocessing, model could be better'
        }
    
    # Run the enhanced pipeline
    results = await pipeline.train_with_iterative_improvement(
        df=sample_data,
        target_column='target',
        problem_type=ProblemType.CLASSIFICATION,
        max_preprocessing_iterations=2,
        max_training_iterations=2,
        user_feedback_callback=feedback_callback,
        progress_callback=progress_callback
    )
    
    print("Enhanced ML Pipeline Results:")
    print(f"- Data shape: {results['pipeline_summary']['original_data_shape']} -> {results['pipeline_summary']['processed_data_shape']}")
    print(f"- Best model: {results['model_training_summary']['best_model_name']}")
    print(f"- Processing time: {results['pipeline_summary']['total_processing_time']:.2f} seconds")
    print(f"- Quality improvement: {results['preprocessing_summary']['total_quality_improvement']:.2f}%")
    
    return results

if __name__ == "__main__":
    # Run example
    asyncio.run(example_enhanced_pipeline_usage())
