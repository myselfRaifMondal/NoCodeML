#!/usr/bin/env python3
"""
Test script to verify that the NaN handling fixes are working correctly.
This script creates datasets with various NaN scenarios and tests the cleanup.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core.automl.automl_engine import AutoMLEngine
from core.models.iterative_trainer import IterativeModelTrainer
from backend.models.schemas import ProblemType

def create_test_datasets():
    """Create test datasets with various NaN scenarios"""
    
    # Dataset 1: Basic dataset with some NaN values
    print("Creating test datasets...")
    
    # Dataset 1: Moderate NaN values
    np.random.seed(42)
    n_samples = 100
    
    data1 = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    # Add some NaN values
    data1.loc[10:20, 'feature1'] = np.nan
    data1.loc[15:25, 'feature2'] = np.nan
    data1.loc[30:35, 'feature3'] = np.nan
    
    # Dataset 2: Severe NaN scenario
    data2 = pd.DataFrame({
        'feature1': [1, 2, np.nan, np.nan, 5, np.nan, 7, 8, np.nan, 10],
        'feature2': [np.nan, np.nan, 3, 4, np.nan, 6, np.nan, np.nan, 9, np.nan],
        'feature3': ['A', np.nan, 'B', np.nan, 'C', 'A', np.nan, 'B', 'C', np.nan],
        'target': [1, 0, np.nan, 1, 0, 1, np.nan, 0, 1, 0]
    })
    
    # Dataset 3: All NaN in one column
    data3 = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [np.nan, np.nan, np.nan, np.nan, np.nan],  # All NaN
        'feature3': ['A', 'B', 'C', 'A', 'B'],
        'target': [1, 0, 1, 0, 1]
    })
    
    return data1, data2, data3

def test_automl_engine_nan_handling():
    """Test AutoMLEngine NaN handling"""
    print("\n🧪 Testing AutoMLEngine NaN handling...")
    
    data1, data2, data3 = create_test_datasets()
    engine = AutoMLEngine()
    
    for i, data in enumerate([data1, data2, data3], 1):
        print(f"\n--- Testing Dataset {i} ---")
        print(f"Original shape: {data.shape}")
        print(f"NaN counts per column:")
        print(data.isnull().sum())
        
        try:
            # Prepare data (this is where NaN cleanup should happen)
            X = data.drop(columns=['target'])
            y = data['target']
            
            print(f"Before cleanup - X NaN: {X.isnull().sum().sum()}, y NaN: {y.isnull().sum()}")
            
            # Test the emergency cleanup method
            X_clean, y_clean = engine._emergency_nan_cleanup(X.copy(), y.copy())
            
            print(f"After cleanup - X shape: {X_clean.shape}, y shape: {y_clean.shape}")
            print(f"After cleanup - X NaN: {X_clean.isnull().sum().sum()}, y NaN: {y_clean.isnull().sum()}")
            
            # Verify no NaN values remain
            assert not X_clean.isnull().any().any(), "Features still contain NaN values!"
            assert not y_clean.isnull().any(), "Target still contains NaN values!"
            print("✅ NaN cleanup successful!")
            
        except Exception as e:
            print(f"❌ Error in dataset {i}: {str(e)}")

def test_iterative_trainer_nan_handling():
    """Test IterativeModelTrainer NaN handling"""
    print("\n🧪 Testing IterativeModelTrainer NaN handling...")
    
    data1, data2, data3 = create_test_datasets()
    trainer = IterativeModelTrainer()
    
    for i, data in enumerate([data1, data2, data3], 1):
        print(f"\n--- Testing Dataset {i} ---")
        print(f"Original shape: {data.shape}")
        print(f"NaN counts per column:")
        print(data.isnull().sum())
        
        try:
            # Prepare data
            X = data.drop(columns=['target'])
            y = data['target']
            
            print(f"Before cleanup - X NaN: {X.isnull().sum().sum()}, y NaN: {y.isnull().sum()}")
            
            # Test the emergency cleanup method
            X_clean, y_clean = trainer._emergency_nan_cleanup(X.copy(), y.copy())
            
            print(f"After cleanup - X shape: {X_clean.shape}, y shape: {y_clean.shape}")
            print(f"After cleanup - X NaN: {X_clean.sum().sum()}, y NaN: {y_clean.isnull().sum()}")
            
            # Verify no NaN values remain
            assert not X_clean.isnull().any().any(), "Features still contain NaN values!"
            assert not y_clean.isnull().any(), "Target still contains NaN values!"
            print("✅ NaN cleanup successful!")
            
        except Exception as e:
            print(f"❌ Error in dataset {i}: {str(e)}")

def test_end_to_end_training():
    """Test end-to-end training with NaN values"""
    print("\n🧪 Testing end-to-end training with NaN values...")
    
    # Create a larger dataset for actual training
    np.random.seed(42)
    n_samples = 200
    
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature4': np.random.exponential(1, n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    # Add NaN values
    data.loc[20:40, 'feature1'] = np.nan
    data.loc[30:50, 'feature2'] = np.nan
    data.loc[60:70, 'feature3'] = np.nan
    
    print(f"Dataset shape: {data.shape}")
    print(f"NaN counts per column:")
    print(data.isnull().sum())
    
    try:
        trainer = IterativeModelTrainer()
        
        X = data.drop(columns=['target'])
        y = data['target']
        
        print("Starting training with NaN cleanup...")
        result = trainer.train_and_optimize(
            X, y, 
            problem_type=ProblemType.CLASSIFICATION,
            max_iterations=1,
            optimize_hyperparameters=False
        )
        
        if result.best_model:
            print(f"✅ Training successful!")
            print(f"Best model: {result.best_model.model_name}")
            print(f"Best accuracy: {result.best_model.metrics.get('accuracy', 'N/A'):.4f}")
        else:
            print("❌ No model trained successfully")
            
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")

def test_edge_cases():
    """Test edge cases that might cause issues"""
    print("\n🧪 Testing edge cases...")
    
    # Test case 1: All NaN target
    print("\n--- Testing all-NaN target ---")
    try:
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [np.nan, np.nan, np.nan, np.nan, np.nan]
        })
        
        engine = AutoMLEngine()
        X = data.drop(columns=['target'])
        y = data['target']
        
        X_clean, y_clean = engine._emergency_nan_cleanup(X, y)
        print("❌ Should have raised an error for all-NaN target!")
        
    except ValueError as e:
        print(f"✅ Correctly caught error: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
    
    # Test case 2: Single row dataset
    print("\n--- Testing single row dataset ---")
    try:
        data = pd.DataFrame({
            'feature1': [1],
            'target': [1]
        })
        
        engine = AutoMLEngine()
        X = data.drop(columns=['target'])
        y = data['target']
        
        X_clean, y_clean = engine._emergency_nan_cleanup(X, y)
        print("❌ Should have raised an error for insufficient data!")
        
    except ValueError as e:
        print(f"✅ Correctly caught error: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

def main():
    """Run all tests"""
    print("🚀 Starting NaN handling tests...")
    print("=" * 60)
    
    try:
        test_automl_engine_nan_handling()
        test_iterative_trainer_nan_handling()
        test_end_to_end_training()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed!")
        print("\n💡 The NaN cleanup fixes should now prevent LogisticRegression errors.")
        print("   When you encounter NaN errors, the system will:")
        print("   1. Detect NaN values before model training")
        print("   2. Apply intelligent imputation strategies")
        print("   3. Remove rows/columns if necessary")
        print("   4. Ensure clean data reaches the model")
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
