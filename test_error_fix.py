#!/usr/bin/env python3
"""
Test script to verify that the comprehensive error handler
fixes the class imbalance issue that was causing:
"Pipeline failed: The least populated class in y has only 1 member"
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'core' / 'utils'))

try:
    from core.utils.comprehensive_error_handler import ComprehensiveErrorHandler
    from backend.models.schemas import ProblemType
    from core.automl.automl_engine import AutoMLEngine
    print("✅ Successfully imported comprehensive error handler and AutoML engine")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def create_problematic_dataset():
    """Create a dataset that would previously cause the class imbalance error"""
    print("🔧 Creating problematic dataset with class imbalance...")
    
    # Create a dataset where some classes have only 1 sample
    np.random.seed(42)
    
    # Create features
    n_samples = 20
    n_features = 5
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create highly imbalanced target with single-sample classes
    y = pd.Series([
        'class_A', 'class_A', 'class_A', 'class_A', 'class_A',  # 5 samples
        'class_B', 'class_B', 'class_B',  # 3 samples
        'class_C', 'class_C',  # 2 samples
        'class_D',  # 1 sample (this would cause the error)
        'class_E',  # 1 sample (this would cause the error)
        'class_A', 'class_A', 'class_A',  # More class A
        'class_B', 'class_B',  # More class B
        'class_F',  # 1 sample (this would cause the error)
        'class_G',  # 1 sample (this would cause the error)
        'class_H'   # 1 sample (this would cause the error)
    ])
    
    print(f"📊 Dataset created with shape: {X.shape}")
    print(f"📊 Class distribution:\n{y.value_counts()}")
    
    return X, y

def test_without_error_handler():
    """Test the original behavior that would fail"""
    print("\n🧪 Testing original behavior (should fail)...")
    
    X, y = create_problematic_dataset()
    
    try:
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f"❌ Unexpected success: {scores}")
        
    except ValueError as e:
        if "least populated class" in str(e) or "minimum number of groups" in str(e):
            print(f"✅ Expected error caught: {e}")
            return True
        else:
            print(f"❌ Unexpected error: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error type: {type(e).__name__}: {e}")
        return False

def test_with_error_handler():
    """Test with comprehensive error handler (should work)"""
    print("\n🛠️ Testing with comprehensive error handler...")
    
    X, y = create_problematic_dataset()
    
    try:
        # Initialize error handler
        error_handler = ComprehensiveErrorHandler()
        
        # Auto-fix the dataset
        print("🔧 Applying auto-fix to dataset...")
        X_fixed, y_fixed = error_handler.auto_fix_dataset(X, y, ProblemType.CLASSIFICATION)
        
        print(f"📊 Fixed dataset shape: {X_fixed.shape}")
        print(f"📊 Fixed class distribution:\n{y_fixed.value_counts()}")
        
        # Test safe cross-validation
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
        
        print("🔧 Testing safe cross-validation...")
        cv_scores = error_handler.safe_cross_validation(model, X_fixed, y_fixed, cv=5)
        
        print(f"✅ Safe cross-validation succeeded!")
        print(f"📊 CV Scores: {cv_scores}")
        print(f"📊 Mean CV Score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        # Test model training
        print("🔧 Testing model training...")
        model.fit(X_fixed, y_fixed)
        print("✅ Model training succeeded!")
        
        # Get error report
        error_report = error_handler.get_error_report()
        print(f"\n📋 Error Handler Report:")
        print(f"   - Errors fixed: {error_report['total_fixes_applied']}")
        print(f"   - Fallback activated: {error_report['fallback_activated']}")
        if error_report['errors_fixed']:
            print(f"   - Fixes applied:")
            for fix in error_report['errors_fixed']:
                print(f"     • {fix}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handler test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_automl_integration():
    """Test the AutoML engine with problematic data"""
    print("\n🚀 Testing AutoML engine integration...")
    
    # Create test dataset file
    X, y = create_problematic_dataset()
    df = X.copy()
    df['target'] = y
    
    test_file = project_root / 'test_imbalanced_data.csv'
    df.to_csv(test_file, index=False)
    print(f"📁 Created test dataset: {test_file}")
    
    try:
        # Test AutoML engine
        automl = AutoMLEngine()
        
        config = {
            'target_column': 'target',
            'feature_columns': [f'feature_{i}' for i in range(5)],
            'problem_type': 'classification',
            'algorithm': 'RandomForest',
            'model_id': 'test_imbalanced'
        }
        
        print("🔧 Training model with AutoML engine...")
        
        # Mock progress callback
        async def mock_progress(message, progress):
            print(f"   📈 [{progress:3d}%] {message}")
        
        # This should work now with our error handling
        import asyncio
        result = asyncio.run(automl.train_model(str(test_file), config, mock_progress))
        
        print("✅ AutoML training succeeded!")
        print(f"📊 Model: {result['algorithm']}")
        print(f"📊 Metrics: {result['metrics']}")
        print(f"📊 CV Scores: {result['cv_scores']}")
        
        # Cleanup
        test_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ AutoML test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup
        if test_file.exists():
            test_file.unlink()
        
        return False

def main():
    """Run all tests"""
    print("🎯 Testing Comprehensive Error Handler for Class Imbalance Fix")
    print("=" * 70)
    
    # Test 1: Verify the original error exists
    test1_passed = test_without_error_handler()
    
    # Test 2: Verify our error handler fixes it
    test2_passed = test_with_error_handler()
    
    # Test 3: Verify AutoML integration works
    test3_passed = test_automl_integration()
    
    print("\n" + "=" * 70)
    print("📋 TEST RESULTS:")
    print(f"   1. Original error detection: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   2. Error handler fix: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"   3. AutoML integration: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    
    overall_success = test1_passed and test2_passed and test3_passed
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED! The class imbalance error has been fixed!")
        print("🚀 Your NoCodeML pipeline should now handle any dataset automatically!")
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
