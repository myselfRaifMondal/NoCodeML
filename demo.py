#!/usr/bin/env python3
"""
NoCodeML Demo Script

This script demonstrates the complete NoCodeML workflow:
1. Upload and analyze a dataset
2. Get intelligent recommendations
3. Train a model with real AutoML
4. Evaluate and make predictions

Run this after starting the server to see NoCodeML in action!
"""

import requests
import json
import time
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
SAMPLE_DATASET = "data/samples/titanic_sample.csv"

def upload_dataset():
    """Upload the sample dataset"""
    print("🔄 Uploading sample dataset...")
    
    url = f"{BASE_URL}/api/dataset/upload"
    
    with open(SAMPLE_DATASET, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        dataset_id = result['dataset_id']
        dataset_info = result['dataset_info']
        
        print("✅ Dataset uploaded successfully!")
        print(f"📊 Dataset ID: {dataset_id}")
        print(f"📈 Rows: {dataset_info['rows']}, Columns: {dataset_info['columns']}")
        print(f"🎯 Data Quality Score: {dataset_info['data_quality_score']}/100")
        print(f"🤖 Recommended Problem Types: {', '.join(dataset_info['recommended_problem_types'])}")
        
        if dataset_info['suggestions']:
            print("💡 Suggestions:")
            for suggestion in dataset_info['suggestions']:
                print(f"   • {suggestion}")
        
        if dataset_info['warnings']:
            print("⚠️  Warnings:")
            for warning in dataset_info['warnings']:
                print(f"   • {warning}")
        
        return dataset_id
    else:
        print(f"❌ Upload failed: {response.text}")
        return None

def get_recommendations(dataset_id):
    """Get intelligent model recommendations"""
    print(f"\n🧠 Getting intelligent recommendations for dataset {dataset_id}...")
    
    url = f"{BASE_URL}/api/analysis/recommend/{dataset_id}"
    payload = {
        "problem_type": "classification",
        "target_column": "Survived",
        "feature_columns": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
        "test_size": 0.2,
        "cv_folds": 5,
        "max_training_time": 300
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        
        print("✅ Recommendations generated!")
        print(f"🎯 Problem Type: {result['problem_type']}")
        print(f"🎯 Target Column: {result['target_column']}")
        print(f"📊 Feature Columns: {', '.join(result['feature_columns'])}")
        print(f"⏱️  Estimated Training Time: {result['estimated_training_time']} minutes")
        print(f"🔧 Preprocessing Required: {'Yes' if result['data_preprocessing_required'] else 'No'}")
        
        print(f"\n🏆 Top Recommended Models:")
        for i, model in enumerate(result['recommended_models'], 1):
            print(f"\n{i}. {model['algorithm']} ({model['algorithm_type']})")
            print(f"   Confidence: {model['confidence_score']:.2f}")
            print(f"   Expected Performance: {model['expected_performance']}")
            print(f"   Preprocessing: {', '.join(model['preprocessing_steps'])}")
            print(f"   Explanation: {model['explanation']}")
            print(f"   Hyperparameters: {model['hyperparameters']}")
        
        return result
    else:
        print(f"❌ Recommendations failed: {response.text}")
        return None

def train_model(dataset_id, recommendations):
    """Train the top recommended model"""
    if not recommendations or not recommendations['recommended_models']:
        print("❌ No recommendations available for training")
        return None
    
    top_model = recommendations['recommended_models'][0]
    
    print(f"\n🚀 Training {top_model['algorithm']} model...")
    
    url = f"{BASE_URL}/api/model/train"
    payload = {
        "dataset_id": dataset_id,
        "model_name": f"NoCodeML {top_model['algorithm']} Model",
        "algorithm": top_model['algorithm'],
        "hyperparameters": top_model['hyperparameters'],
        "problem_type": "classification",
        "target_column": recommendations['target_column'],
        "feature_columns": recommendations['feature_columns'],
        "preprocessing_config": {},
        "validation_config": {}
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        model_id = result['model_id']
        
        print("✅ Training started!")
        print(f"🆔 Model ID: {model_id}")
        
        # Monitor training progress
        return monitor_training(model_id)
    else:
        print(f"❌ Training failed: {response.text}")
        return None

def monitor_training(model_id):
    """Monitor training progress"""
    print("\n📊 Monitoring training progress...")
    
    while True:
        url = f"{BASE_URL}/api/model/training/{model_id}/status"
        response = requests.get(url)
        
        if response.status_code == 200:
            status = response.json()
            
            print(f"\r🔄 {status['current_step']} ({status['progress_percentage']:.1f}%)", end="", flush=True)
            
            if status['status'] == 'completed':
                print(f"\n✅ Training completed successfully!")
                return model_id
            elif status['status'] == 'failed':
                print(f"\n❌ Training failed!")
                return None
            
            time.sleep(2)
        else:
            print(f"\n❌ Failed to get training status: {response.text}")
            return None

def evaluate_model(model_id):
    """Get detailed model evaluation"""
    print(f"\n📈 Evaluating model {model_id}...")
    
    # Get model details
    url = f"{BASE_URL}/api/model/{model_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        model = response.json()
        
        print("✅ Model evaluation complete!")
        print(f"🤖 Algorithm: {model['algorithm']}")
        print(f"⏱️  Training Time: {model['training_time']} seconds")
        print(f"💾 Model Size: {model['model_size_mb']} MB")
        
        if model['metrics']:
            print(f"\n📊 Performance Metrics:")
            for metric, value in model['metrics'].items():
                if value is not None:
                    print(f"   • {metric.replace('_', ' ').title()}: {value:.4f}")
        
        if model['feature_importance']:
            print(f"\n🎯 Top Feature Importance:")
            sorted_features = sorted(model['feature_importance'].items(), 
                                   key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:5]:
                print(f"   • {feature}: {importance:.4f}")
        
        print(f"\n📝 Model Explanation:")
        print(f"   {model['model_explanation']}")
        
        return model
    else:
        print(f"❌ Model evaluation failed: {response.text}")
        return None

def make_prediction(model_id):
    """Make a sample prediction"""
    print(f"\n🔮 Making sample predictions with model {model_id}...")
    
    # Sample passenger data
    sample_data = {
        "Pclass": 3,
        "Sex": "male", 
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }
    
    url = f"{BASE_URL}/api/model/predict"
    payload = {
        "model_id": model_id,
        "input_data": sample_data,
        "return_probabilities": True,
        "explain_prediction": True
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        
        print("✅ Prediction complete!")
        print(f"🎯 Input: {sample_data}")
        print(f"📊 Prediction: {result['predictions']}")
        
        if result.get('probabilities'):
            print(f"📈 Probabilities: {result['probabilities']}")
        
        if result.get('explanations'):
            print(f"💡 Explanation: Feature importance contributed to this prediction")
        
        return result
    else:
        print(f"❌ Prediction failed: {response.text}")
        return None

def main():
    """Run the complete NoCodeML demo"""
    print("🚀 NoCodeML Complete Demo")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Server is not running! Please start with: python -m uvicorn backend.main:app --reload")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running! Please start with: python -m uvicorn backend.main:app --reload")
        return
    
    print("✅ Server is running!")
    
    # Step 1: Upload dataset
    dataset_id = upload_dataset()
    if not dataset_id:
        return
    
    # Step 2: Get recommendations
    recommendations = get_recommendations(dataset_id)
    if not recommendations:
        return
    
    # Step 3: Train model
    model_id = train_model(dataset_id, recommendations)
    if not model_id:
        return
    
    # Step 4: Evaluate model
    model = evaluate_model(model_id)
    if not model:
        return
    
    # Step 5: Make predictions
    prediction = make_prediction(model_id)
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📊 Dataset analyzed, model trained, and predictions made!")
    print(f"🔗 Visit {BASE_URL}/api/docs for full API documentation")

if __name__ == "__main__":
    main()

