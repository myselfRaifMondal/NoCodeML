#!/usr/bin/env python3
"""
NoCodeML UI Demo Walkthrough

This script demonstrates the key features of the NoCodeML user interface
by simulating the workflow that non-technical users will experience.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def demonstrate_data_analysis():
    """Demonstrate the data analysis capabilities"""
    print("🔍 STEP 1: DATA ANALYSIS DEMO")
    print("=" * 50)
    
    # Load sample data
    print("📂 Loading sample house prices dataset...")
    df = pd.read_csv('sample_data/house_prices_sample.csv')
    
    print(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    print("\n📊 DATASET PREVIEW:")
    print(df.head())
    
    print("\n📈 DATASET OVERVIEW:")
    print(f"• Total Rows: {len(df)}")
    print(f"• Total Columns: {len(df.columns)}")
    print(f"• Missing Values: {df.isnull().sum().sum()}")
    print(f"• Duplicate Rows: {df.duplicated().sum()}")
    
    # Calculate quality score (same logic as in the UI)
    missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    duplicate_count = df.duplicated().sum()
    quality_score = max(0, 100 - missing_percentage - (duplicate_count / len(df) * 10))
    print(f"• Data Quality Score: {quality_score:.1f}%")
    
    print("\n📋 COLUMN ANALYSIS:")
    for col in df.columns:
        col_data = df[col]
        data_type = "numeric" if pd.api.types.is_numeric_dtype(col_data) else "categorical"
        unique_count = col_data.nunique()
        missing_count = col_data.isnull().sum()
        
        print(f"  • {col}: {data_type}, {unique_count} unique values, {missing_count} missing")
    
    return df

def demonstrate_model_building(df):
    """Demonstrate the model building process"""
    print("\n\n🤖 STEP 2: MODEL BUILDING DEMO")
    print("=" * 50)
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import LabelEncoder
    
    print("🎯 Problem Type: REGRESSION (predicting house prices)")
    print("🎯 Target Variable: 'price'")
    
    # Prepare features and target
    target_column = 'price'
    feature_columns = [col for col in df.columns if col != target_column]
    print(f"🎯 Features Selected: {feature_columns}")
    
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    # Handle categorical variables
    print("\n🔄 Processing categorical variables...")
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            print(f"  • Encoding '{col}' (categorical)")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Split data
    print("\n✂️ Splitting data: 80% training, 20% testing")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    print("\n🚂 Training models...")
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    for name, model in models.items():
        print(f"  • Training {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        results[name] = {
            'model': model,
            'mse': mse,
            'r2': r2,
            'predictions': predictions
        }
        
        print(f"    ✅ {name}: R² = {r2:.3f}, MSE = {mse:.0f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_r2 = results[best_model_name]['r2']
    
    print(f"\n🏆 Best Model: {best_model_name} (R² Score: {best_r2:.3f})")
    
    return results, X_test, y_test, label_encoders, feature_columns

def demonstrate_predictions(results, X_test, feature_columns):
    """Demonstrate making predictions"""
    print("\n\n🔮 STEP 3: MAKING PREDICTIONS DEMO")
    print("=" * 50)
    
    # Get best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model = results[best_model_name]['model']
    
    print(f"🎯 Using best model: {best_model_name}")
    
    # Make a prediction with sample data
    sample_house = X_test.iloc[0:1].copy()
    prediction = best_model.predict(sample_house)[0]
    
    print("\n🏠 SAMPLE PREDICTION:")
    print("Input features:")
    for i, col in enumerate(feature_columns):
        value = sample_house.iloc[0, i]
        if col == 'location':
            # Decode location if it was encoded
            locations = ['rural', 'suburban', 'urban']
            value = locations[int(value)] if value < len(locations) else f"code_{int(value)}"
        print(f"  • {col}: {value}")
    
    print(f"\n💰 Predicted Price: ${prediction:,.0f}")
    
    return prediction

def demonstrate_feature_importance(results):
    """Demonstrate feature importance analysis"""
    print("\n\n🔍 STEP 4: FEATURE IMPORTANCE DEMO")
    print("=" * 50)
    
    # Get Random Forest model for feature importance
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        
        if hasattr(rf_model, 'feature_importances_'):
            print("📊 Feature Importance (Random Forest):")
            feature_names = ['bedrooms', 'bathrooms', 'square_feet', 'age_years', 'garage', 'location']
            
            importance_data = list(zip(feature_names, rf_model.feature_importances_))
            importance_data.sort(key=lambda x: x[1], reverse=True)
            
            for feature, importance in importance_data:
                bar_length = int(importance * 50)  # Scale for visual bar
                bar = "█" * bar_length
                print(f"  • {feature:12}: {importance:.3f} {bar}")
    
def demonstrate_classification_example():
    """Demonstrate classification workflow with customer churn data"""
    print("\n\n🎯 BONUS: CLASSIFICATION EXAMPLE")
    print("=" * 50)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    
    # Load customer churn data
    print("📂 Loading customer churn dataset...")
    df = pd.read_csv('sample_data/customer_churn_sample.csv')
    
    print(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    print("🎯 Problem Type: CLASSIFICATION (predicting customer churn)")
    
    # Prepare data
    target_column = 'churn'
    feature_columns = ['age', 'monthly_bill', 'total_usage_gb', 'customer_service_calls', 'contract_length', 'payment_method']
    
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    # Encode categorical variables
    print("\n🔄 Processing categorical variables...")
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y.astype(str))
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n🚂 Training classification models...")
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"  ✅ {name}: Accuracy = {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    print("\n🎯 Classification Result: Models can predict customer churn with good accuracy!")

def main():
    """Main demo function"""
    print("🤖 NoCodeML UI FUNCTIONALITY DEMONSTRATION")
    print("=" * 60)
    print()
    print("This demo shows what happens behind the scenes when users")
    print("interact with the NoCodeML web interface.")
    print()
    print("👤 USER WORKFLOW:")
    print("1. Upload data → 2. Analyze → 3. Build model → 4. Make predictions")
    print()
    
    try:
        # Demo regression workflow
        df = demonstrate_data_analysis()
        results, X_test, y_test, label_encoders, feature_columns = demonstrate_model_building(df)
        demonstrate_predictions(results, X_test, feature_columns)
        demonstrate_feature_importance(results)
        
        # Demo classification workflow
        demonstrate_classification_example()
        
        print("\n\n🎉 DEMO COMPLETE!")
        print("=" * 50)
        print("✅ The NoCodeML UI provides all this functionality")
        print("✅ Non-technical users can do this through point-and-click")
        print("✅ No coding knowledge required!")
        print()
        print("🚀 To try the actual web interface:")
        print("   source venv/bin/activate")
        print("   python run_ui.py")
        print()
        print("🌐 Then visit: http://localhost:8501")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("Make sure sample datasets are available in sample_data/")

if __name__ == "__main__":
    main()
