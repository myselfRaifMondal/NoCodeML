#!/usr/bin/env python3
"""
Demo Script for Enhanced Data Preprocessing and Iterative Improvement

This script demonstrates the advanced preprocessing capabilities including:
- Comprehensive data quality assessment
- Intelligent missing data handling
- Iterative improvement with model feedback
- User feedback integration
- Issue detection and resolution
"""

import pandas as pd
import numpy as np
import asyncio
from datetime import datetime
from pathlib import Path
import logging
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core.integration.preprocessing_trainer_integration import EnhancedMLPipeline
from core.preprocessing.enhanced_preprocessor import EnhancedDataPreprocessor
from backend.models.schemas import ProblemType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_demo_dataset():
    """Create a demonstration dataset with various data quality issues"""
    logger.info("Creating demonstration dataset with data quality issues...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create base data
    data = {
        # Numeric features with various issues
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.exponential(50000, n_samples),  # Skewed distribution
        'credit_score': np.random.normal(650, 100, n_samples),
        'years_employed': np.random.poisson(5, n_samples),
        
        # Categorical features
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'job_category': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Education', 'Other'], n_samples),
        'city': np.random.choice([f'City_{i}' for i in range(50)], n_samples),  # High cardinality
        
        # Features with correlation issues
        'monthly_income': None,  # Will be derived from income (correlation issue)
        'debt_ratio': np.random.uniform(0, 1, n_samples),
        
        # Target variable
        'approved': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    }
    
    # Create correlated feature
    data['monthly_income'] = data['income'] / 12 + np.random.normal(0, 100, n_samples)
    
    # Create perfect correlation (data leakage)
    data['approval_indicator'] = data['approved']  # Perfect correlation with target
    
    df = pd.DataFrame(data)
    
    # Introduce missing values in different patterns
    # Random missing values
    missing_indices = np.random.choice(df.index, size=int(0.1 * n_samples), replace=False)
    df.loc[missing_indices, 'credit_score'] = np.nan
    
    # Pattern-based missing values (income missing when unemployed)
    unemployed_mask = df['years_employed'] == 0
    df.loc[unemployed_mask, 'income'] = np.nan
    df.loc[unemployed_mask, 'monthly_income'] = np.nan
    
    # High missing percentage column
    high_missing_indices = np.random.choice(df.index, size=int(0.8 * n_samples), replace=False)
    df.loc[high_missing_indices, 'debt_ratio'] = np.nan
    
    # Add duplicate rows
    duplicate_rows = df.sample(n=50, random_state=42)
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    # Add outliers
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[outlier_indices, 'age'] = np.random.uniform(150, 200, size=20)  # Impossible ages
    df.loc[outlier_indices[:10], 'income'] = np.random.uniform(10000000, 50000000, size=10)  # Extreme incomes
    
    # Add inconsistent data types
    df.loc[df.index[:10], 'age'] = df.loc[df.index[:10], 'age'].astype(str)
    
    logger.info(f"Demo dataset created with {df.shape[0]} rows and {df.shape[1]} columns")
    logger.info(f"Missing data: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%")
    logger.info(f"Duplicate rows: {df.duplicated().sum()}")
    
    return df

async def demo_basic_preprocessing():
    """Demonstrate basic preprocessing capabilities"""
    print("\n" + "="*80)
    print("🧹 DEMO: Basic Data Preprocessing")
    print("="*80)
    
    # Create demo dataset
    df = create_demo_dataset()
    
    # Initialize preprocessor
    preprocessor = EnhancedDataPreprocessor(learning_enabled=True)
    
    print(f"\n📊 Original Dataset Shape: {df.shape}")
    print(f"📊 Original Missing Data: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%")
    print(f"📊 Original Duplicates: {df.duplicated().sum()}")
    
    # Perform comprehensive preprocessing
    print("\n🔄 Running comprehensive preprocessing...")
    processed_df, result = preprocessor.comprehensive_preprocessing(
        df, 
        target_column='approved',
        problem_type=ProblemType.CLASSIFICATION
    )
    
    print(f"\n✅ Preprocessing completed!")
    print(f"📊 Processed Dataset Shape: {processed_df.shape}")
    print(f"📊 Processed Missing Data: {(processed_df.isnull().sum().sum() / (processed_df.shape[0] * processed_df.shape[1]) * 100):.1f}%")
    print(f"📊 Quality Improvement: {result.quality_improvement:.2f}%")
    print(f"🔧 Transformations Applied: {len(result.transformations_applied)}")
    print(f"⚠️ Issues Found: {len(result.issues_found)}")
    print(f"✅ Issues Fixed: {len(result.issues_fixed)}")
    
    # Display some transformations
    print("\n🔧 Sample Transformations Applied:")
    for i, transformation in enumerate(result.transformations_applied[:5], 1):
        print(f"  {i}. {transformation}")
    
    # Display data quality metrics
    print("\n📈 Quality Metrics:")
    print(f"  • Data Quality Score: {result.metrics.data_quality_score:.1f}%")
    print(f"  • Feature Quality Score: {result.metrics.feature_quality_score:.1f}%")
    print(f"  • Processing Time: {result.metrics.preprocessing_time:.2f} seconds")
    
    return processed_df, result

async def demo_iterative_improvement():
    """Demonstrate iterative improvement with model feedback"""
    print("\n" + "="*80)
    print("🚀 DEMO: Iterative Improvement with Model Feedback")
    print("="*80)
    
    # Create demo dataset
    df = create_demo_dataset()
    
    # Initialize enhanced pipeline
    pipeline = EnhancedMLPipeline(learning_enabled=True)
    
    # Progress callback
    async def progress_callback(message: str, progress: int):
        print(f"🔄 [{progress:3d}%] {message}")
    
    # User feedback callback (simulated)
    async def feedback_callback(results):
        print("\n📝 Simulating user feedback...")
        return {
            'overall_satisfaction': 8,
            'preprocessing_satisfaction': 9,
            'model_satisfaction': 7,
            'ease_of_use': 9,
            'comments': 'Great preprocessing! Model performance could be improved.',
            'timestamp': datetime.now().isoformat()
        }
    
    print(f"\n📊 Starting Enhanced ML Pipeline...")
    print(f"📊 Original Dataset Shape: {df.shape}")
    
    # Run iterative improvement
    results = await pipeline.train_with_iterative_improvement(
        df=df,
        target_column='approved',
        problem_type=ProblemType.CLASSIFICATION,
        max_preprocessing_iterations=3,
        max_training_iterations=2,
        improvement_threshold=0.01,
        user_feedback_callback=feedback_callback,
        progress_callback=progress_callback
    )
    
    # Display comprehensive results
    print("\n" + "="*60)
    print("📈 COMPREHENSIVE RESULTS")
    print("="*60)
    
    # Pipeline summary
    summary = results['pipeline_summary']
    print(f"\n⏱️ Total Processing Time: {summary['total_processing_time']:.2f} seconds")
    print(f"🔄 Preprocessing Iterations: {summary['preprocessing_iterations']}")
    print(f"🤖 Model Training Iterations: {summary['model_training_iterations']}")
    print(f"📊 Data Shape Change: {summary['original_data_shape']} → {summary['processed_data_shape']}")
    
    # Preprocessing summary
    prep_summary = results['preprocessing_summary']
    print(f"\n🧹 Preprocessing Results:")
    print(f"  • Quality Improvement: {prep_summary['total_quality_improvement']:.2f}%")
    print(f"  • Issues Found: {prep_summary['issues_found']}")
    print(f"  • Issues Fixed: {prep_summary['issues_fixed']}")
    print(f"  • Transformations Applied: {prep_summary['transformations_applied']}")
    print(f"  • Data Quality Score: {prep_summary['data_quality_score']:.1f}%")
    print(f"  • Feature Quality Score: {prep_summary['feature_quality_score']:.1f}%")
    
    # Model training summary
    model_summary = results['model_training_summary']
    print(f"\n🤖 Model Training Results:")
    print(f"  • Best Model: {model_summary['best_model_name']}")
    print(f"  • Models Trained: {model_summary['total_models_trained']}")
    
    if model_summary['best_model_performance']:
        print(f"  • Model Performance:")
        for metric, value in model_summary['best_model_performance'].items():
            print(f"    - {metric}: {value:.4f}")
    
    # Recommendations
    if results['recommendations']:
        print(f"\n💡 Top Recommendations:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
    
    # Next steps
    if results['next_steps']:
        print(f"\n📋 Next Steps:")
        for step in results['next_steps'][:3]:
            print(f"  • {step}")
    
    return results

async def demo_quality_assessment():
    """Demonstrate comprehensive data quality assessment"""
    print("\n" + "="*80)
    print("🔍 DEMO: Comprehensive Data Quality Assessment")
    print("="*80)
    
    # Create demo dataset
    df = create_demo_dataset()
    
    # Initialize preprocessor
    preprocessor = EnhancedDataPreprocessor()
    
    print(f"\n📊 Assessing data quality for dataset with {df.shape[0]} rows and {df.shape[1]} columns...")
    
    # Perform quality assessment
    issues = preprocessor._comprehensive_data_assessment(df, 'approved', ProblemType.CLASSIFICATION)
    
    print(f"\n🔍 Quality Assessment Results:")
    print(f"📊 Total Issues Found: {len(issues)}")
    
    # Group issues by severity
    severity_groups = {'CRITICAL': [], 'HIGH': [], 'MEDIUM': [], 'LOW': []}
    for issue in issues:
        severity_groups[issue.severity.value.upper()].append(issue)
    
    # Display issues by severity
    severity_emojis = {'CRITICAL': '🚨', 'HIGH': '⚠️', 'MEDIUM': '💡', 'LOW': 'ℹ️'}
    
    for severity, emoji in severity_emojis.items():
        issues_in_severity = severity_groups[severity]
        if issues_in_severity:
            print(f"\n{emoji} {severity} Issues ({len(issues_in_severity)}):")
            for issue in issues_in_severity:
                print(f"  • {issue.issue_type.value.title()}: {issue.description}")
                print(f"    Solution: {issue.solution}")
                print(f"    Impact: {issue.impact_on_model}")
                if issue.suggested_action:
                    print(f"    Action: {issue.suggested_action}")
                print()
    
    # Generate issue summary statistics
    issue_types = {}
    for issue in issues:
        issue_type = issue.issue_type.value
        if issue_type not in issue_types:
            issue_types[issue_type] = 0
        issue_types[issue_type] += 1
    
    print(f"📈 Issue Type Distribution:")
    for issue_type, count in sorted(issue_types.items()):
        print(f"  • {issue_type.replace('_', ' ').title()}: {count}")
    
    return issues

async def demo_learning_capabilities():
    """Demonstrate learning from user feedback"""
    print("\n" + "="*80)
    print("🧠 DEMO: Learning from User Feedback")
    print("="*80)
    
    # Create demo dataset
    df = create_demo_dataset()
    
    # Initialize preprocessor with learning enabled
    preprocessor = EnhancedDataPreprocessor(learning_enabled=True)
    
    print("🧠 Demonstrating learning capabilities...")
    
    # Simulate multiple preprocessing runs with feedback
    feedback_scenarios = [
        {
            'satisfaction': 6,
            'comments': 'Too aggressive outlier removal',
            'description': 'User unhappy with outlier handling'
        },
        {
            'satisfaction': 8,
            'comments': 'Good missing data handling',
            'description': 'User satisfied with imputation'
        },
        {
            'satisfaction': 9,
            'comments': 'Excellent feature engineering',
            'description': 'User very happy with results'
        }
    ]
    
    for i, scenario in enumerate(feedback_scenarios, 1):
        print(f"\n🔄 Run {i}: {scenario['description']}")
        
        # Run preprocessing
        processed_df, result = preprocessor.comprehensive_preprocessing(
            df.copy(), 
            target_column='approved',
            problem_type=ProblemType.CLASSIFICATION
        )
        
        # Simulate user feedback
        feedback = {
            'overall_satisfaction': scenario['satisfaction'],
            'preprocessing_satisfaction': scenario['satisfaction'],
            'model_satisfaction': scenario['satisfaction'],
            'comments': scenario['comments'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Learn from feedback
        preprocessor._learn_from_feedback(feedback, result)
        
        print(f"  • Quality Improvement: {result.quality_improvement:.2f}%")
        print(f"  • User Satisfaction: {scenario['satisfaction']}/10")
        print(f"  • Feedback: {scenario['comments']}")
        print(f"  • Adaptive Thresholds Updated: ✅")
    
    # Show learned preferences
    print(f"\n🧠 Learned Preferences:")
    print(f"  • Adaptive Thresholds:")
    for threshold, value in preprocessor.adaptive_thresholds.items():
        print(f"    - {threshold}: {value:.3f}")
    
    print(f"  • Strategy Performance Tracking:")
    for strategy_type, strategies in preprocessor.strategy_performance.items():
        if strategies:
            print(f"    - {strategy_type}: {len(strategies)} strategies tracked")
    
    return preprocessor

def display_comparison_table(original_df, processed_df):
    """Display a comparison table of original vs processed data"""
    print("\n" + "="*80)
    print("📊 BEFORE vs AFTER COMPARISON")
    print("="*80)
    
    # Basic statistics
    comparison_data = {
        'Metric': [
            'Rows',
            'Columns', 
            'Missing Values (%)',
            'Duplicate Rows',
            'Numeric Columns',
            'Categorical Columns',
            'Memory Usage (MB)'
        ],
        'Original': [
            f"{original_df.shape[0]:,}",
            f"{original_df.shape[1]:,}",
            f"{(original_df.isnull().sum().sum() / (original_df.shape[0] * original_df.shape[1]) * 100):.1f}%",
            f"{original_df.duplicated().sum():,}",
            f"{len(original_df.select_dtypes(include=['int64', 'float64']).columns):,}",
            f"{len(original_df.select_dtypes(include=['object', 'category']).columns):,}",
            f"{original_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}"
        ],
        'Processed': [
            f"{processed_df.shape[0]:,}",
            f"{processed_df.shape[1]:,}",
            f"{(processed_df.isnull().sum().sum() / (processed_df.shape[0] * processed_df.shape[1]) * 100):.1f}%",
            f"{processed_df.duplicated().sum():,}",
            f"{len(processed_df.select_dtypes(include=['int64', 'float64']).columns):,}",
            f"{len(processed_df.select_dtypes(include=['object', 'category']).columns):,}",
            f"{processed_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display as formatted table
    print(f"{'Metric':<25} {'Original':<15} {'Processed':<15} {'Change':<15}")
    print("-" * 70)
    
    for _, row in comparison_df.iterrows():
        metric = row['Metric']
        original = row['Original']
        processed = row['Processed']
        
        # Calculate change where possible
        try:
            if '%' in original:
                orig_val = float(original.replace('%', '').replace(',', ''))
                proc_val = float(processed.replace('%', '').replace(',', ''))
                change = f"{proc_val - orig_val:+.1f}%"
            elif ',' in original:
                orig_val = int(original.replace(',', ''))
                proc_val = int(processed.replace(',', ''))
                change = f"{proc_val - orig_val:+,}"
            else:
                orig_val = float(original)
                proc_val = float(processed)
                change = f"{proc_val - orig_val:+.2f}"
        except:
            change = "N/A"
        
        print(f"{metric:<25} {original:<15} {processed:<15} {change:<15}")

async def main():
    """Main demo function"""
    print("🤖 NoCodeML Enhanced Preprocessing Demo")
    print("=" * 80)
    print("This demo showcases the advanced data preprocessing capabilities including:")
    print("• Comprehensive data quality assessment")
    print("• Intelligent missing data handling") 
    print("• Iterative improvement with model feedback")
    print("• User feedback integration and learning")
    print("• Issue detection and automatic resolution")
    print("=" * 80)
    
    try:
        # Demo 1: Basic preprocessing
        processed_df, result = await demo_basic_preprocessing()
        
        # Demo 2: Quality assessment
        issues = await demo_quality_assessment()
        
        # Demo 3: Iterative improvement
        comprehensive_results = await demo_iterative_improvement()
        
        # Demo 4: Learning capabilities
        learned_preprocessor = await demo_learning_capabilities()
        
        # Show comparison
        original_df = create_demo_dataset()
        display_comparison_table(original_df, processed_df)
        
        print("\n" + "="*80)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("🎉 The enhanced preprocessing system has demonstrated:")
        print("• Automatic detection and fixing of data quality issues")
        print("• Intelligent preprocessing strategies based on data characteristics")
        print("• Iterative improvement using model performance feedback")
        print("• Learning from user feedback to improve future preprocessing")
        print("• Comprehensive reporting and recommendations")
        print("\n💡 Next Steps:")
        print("• Integrate with your existing ML workflows")
        print("• Customize preprocessing strategies for your domain")
        print("• Use the feedback system to continuously improve results")
        print("• Explore the Streamlit UI for interactive preprocessing")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
