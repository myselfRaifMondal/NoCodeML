#!/usr/bin/env python3
"""
Demo: Using Trained AutoML Model
===============================

This script demonstrates how to load and use a model trained by
the automated ML pipeline to make predictions on new data.

Usage:
    python demo_model_usage.py --model model_file.pkl --data new_data.csv
"""

import pickle
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Import required classes for pickle loading
try:
    from automated_ml_pipeline import AutomatedTransformer
except ImportError:
    print("⚠️  Could not import AutomatedTransformer, some functionality may be limited")
    AutomatedTransformer = None

def load_trained_model(model_path: str):
    """Load the trained model package"""
    print(f"📦 Loading model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    
    print("✅ Model loaded successfully!")
    print(f"   🤖 Algorithm: {model_package['model_name']}")
    print(f"   🎯 Task Type: {'Classification' if model_package['is_classification'] else 'Regression'}")
    print(f"   📊 Features: {len(model_package['feature_names'])}")
    print(f"   📅 Exported: {model_package['export_timestamp']}")
    
    return model_package

def prepare_new_data(data_path: str, model_package: dict):
    """Prepare new data using the same preprocessing pipeline"""
    print(f"\n📋 Loading new data from: {data_path}")
    
    # Load new data
    if data_path.endswith(('.xlsx', '.xls')):
        new_data = pd.read_excel(data_path)
    else:
        new_data = pd.read_csv(data_path)
    
    print(f"   📊 Data shape: {new_data.shape}")
    
    # Extract required components
    feature_names = model_package['feature_names']
    transformer = model_package['transformer']
    
    print(f"   🎯 Required features: {feature_names}")
    
    # Check if all required features are present
    missing_features = set(feature_names) - set(new_data.columns)
    if missing_features:
        print(f"❌ Missing required features: {missing_features}")
        print("   Available features:", list(new_data.columns))
        return None
    
    # Select only required features
    new_data_features = new_data[feature_names].copy()
    print(f"✅ Selected {len(feature_names)} features")
    
    # Apply transformations if transformer is available
    if transformer:
        print("🔧 Applying preprocessing transformations...")
        try:
            # Apply the same transformations used during training
            new_data_transformed, _ = transformer.transform_data(new_data_features)
            print("✅ Data preprocessing completed")
            return new_data_transformed
        except Exception as e:
            print(f"❌ Transformation failed: {e}")
            return None
    else:
        print("⚠️  No transformer found, using raw features")
        return new_data_features

def make_predictions(model_package: dict, prepared_data: pd.DataFrame):
    """Make predictions using the trained model"""
    print("\n🔮 Making predictions...")
    
    model = model_package['model']
    is_classification = model_package['is_classification']
    
    try:
        # Make predictions
        predictions = model.predict(prepared_data)
        print(f"✅ Generated {len(predictions)} predictions")
        
        # For classification, also get probabilities
        if is_classification and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(prepared_data)
            return predictions, probabilities
        else:
            return predictions, None
            
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None, None

def display_results(predictions, probabilities, is_classification: bool):
    """Display prediction results"""
    print("\n📊 Prediction Results:")
    print("=" * 50)
    
    if predictions is None:
        print("❌ No predictions to display")
        return
    
    # Display summary statistics
    if is_classification:
        unique_preds, counts = np.unique(predictions, return_counts=True)
        print("📋 Prediction Summary:")
        for pred, count in zip(unique_preds, counts):
            print(f"   Class {pred}: {count} samples ({count/len(predictions)*100:.1f}%)")
        
        if probabilities is not None:
            print(f"\n🎯 Prediction Confidence (avg): {probabilities.max(axis=1).mean():.3f}")
    else:
        print("📈 Regression Results:")
        print(f"   Mean: {predictions.mean():.3f}")
        print(f"   Std:  {predictions.std():.3f}")
        print(f"   Min:  {predictions.min():.3f}")
        print(f"   Max:  {predictions.max():.3f}")
    
    # Display first few predictions
    print(f"\n🔍 First 10 Predictions:")
    for i in range(min(10, len(predictions))):
        if is_classification and probabilities is not None:
            confidence = probabilities[i].max()
            print(f"   Sample {i+1}: {predictions[i]} (confidence: {confidence:.3f})")
        else:
            print(f"   Sample {i+1}: {predictions[i]}")
    
    if len(predictions) > 10:
        print(f"   ... and {len(predictions)-10} more")

def create_sample_data_for_demo():
    """Create sample data that matches the employee dataset structure"""
    print("📋 Creating sample data for demonstration...")
    
    np.random.seed(42)
    n_samples = 20
    
    sample_data = pd.DataFrame({
        'age': np.random.randint(25, 65, n_samples),
        'income': np.random.normal(55000, 12000, n_samples),
        'education_years': np.random.randint(10, 18, n_samples),
        'experience': np.random.randint(1, 30, n_samples),
        'satisfaction_score': np.random.uniform(3, 9, n_samples)
    })
    
    # Save sample data
    sample_file = "sample_new_data.csv"
    sample_data.to_csv(sample_file, index=False)
    
    print(f"✅ Created {sample_file} with {n_samples} samples")
    print("   Features:", list(sample_data.columns))
    
    return sample_file

def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description="Demo: Using Trained AutoML Model")
    parser.add_argument("--model", help="Path to trained model file (.pkl)")
    parser.add_argument("--data", help="Path to new data file")
    parser.add_argument("--demo", action="store_true", help="Run with sample data")
    
    args = parser.parse_args()
    
    print("🤖 AutoML Model Usage Demo")
    print("=" * 40)
    
    # Find model file if not specified
    if not args.model:
        # Look for recent model files
        model_files = list(Path(".").glob("**/automl_model_*.pkl"))
        if model_files:
            args.model = str(sorted(model_files, key=lambda x: x.stat().st_mtime)[-1])
            print(f"🔍 Using most recent model: {args.model}")
        else:
            print("❌ No model file found. Please specify --model or run the pipeline first.")
            return
    
    # Create sample data if demo mode or no data specified
    if args.demo or not args.data:
        args.data = create_sample_data_for_demo()
    
    try:
        # Load the trained model
        model_package = load_trained_model(args.model)
        
        # Prepare new data
        prepared_data = prepare_new_data(args.data, model_package)
        if prepared_data is None:
            return
        
        # Make predictions
        predictions, probabilities = make_predictions(model_package, prepared_data)
        
        # Display results
        display_results(predictions, probabilities, model_package['is_classification'])
        
        # Save predictions to file
        output_file = "predictions_output.csv"
        results_df = pd.DataFrame({
            'prediction': predictions
        })
        
        if probabilities is not None:
            for i, prob in enumerate(probabilities.T):
                results_df[f'probability_class_{i}'] = prob
        
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Predictions saved to: {output_file}")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    main()
