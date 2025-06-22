#!/usr/bin/env python3
"""
NoCodeML Demo Tutorial

This script demonstrates all the functionality of the NoCodeML platform.
It shows you how to:
1. Upload datasets
2. Analyze data quality
3. Get ML recommendations
4. Interact with the API

Make sure the server is running first: python minimal_server.py
"""

import requests
import json
import time
import pandas as pd
from pathlib import Path
import sys

# Configuration
BASE_URL = "http://localhost:8000"

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def print_step(step_num, description):
    """Print a formatted step"""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 50)

def check_server_health():
    """Check if the server is running"""
    print_header("CHECKING SERVER STATUS")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("✅ Server is healthy!")
            print(f"   Service: {result['service']}")
            print(f"   Version: {result['version']}")
            return True
        else:
            print("❌ Server is not responding correctly")
            return False
    except requests.exceptions.RequestException as e:
        print("❌ Server is not running!")
        print(f"   Error: {e}")
        print("\n💡 To start the server, run:")
        print("   python minimal_server.py")
        return False

def create_sample_datasets():
    """Create sample datasets for demonstration"""
    print_header("CREATING SAMPLE DATASETS")
    
    # Create data directory
    data_dir = Path("data/samples")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset 1: Employee Data (Classification/Regression)
    print("📊 Creating Employee Performance Dataset...")
    employee_data = {
        'Name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson',
                'Diana Lee', 'Frank Miller', 'Grace Taylor', 'Henry Davis', 'Isabel Garcia',
                'Jack White', 'Karen Black', 'Leo Green', 'Mia Blue', 'Noah Red'],
        'Age': [28, 32, 45, 29, 38, 26, 41, 33, 47, 30, 35, 27, 39, 31, 44],
        'Salary': [65000, 78000, 95000, 58000, 82000, 52000, 89000, 71000, 105000, 63000,
                  75000, 56000, 88000, 69000, 98000],
        'Department': ['Engineering', 'Marketing', 'Engineering', 'Sales', 'Marketing',
                      'Sales', 'Engineering', 'Marketing', 'Engineering', 'Sales',
                      'Engineering', 'Sales', 'Marketing', 'Engineering', 'Engineering'],
        'Experience': [3, 5, 10, 2, 7, 1, 9, 6, 12, 4, 8, 2, 11, 5, 13],
        'Performance_Score': [85, 92, 88, 79, 91, 83, 94, 87, 96, 81, 89, 77, 93, 84, 95],
        'Promoted': ['No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No',
                    'Yes', 'No', 'Yes', 'No', 'Yes']
    }
    
    df_employees = pd.DataFrame(employee_data)
    employee_file = data_dir / "employee_performance.csv"
    df_employees.to_csv(employee_file, index=False)
    print(f"   ✅ Created: {employee_file}")
    
    # Dataset 2: Sales Data (Time Series/Regression)
    print("📈 Creating Sales Dataset...")
    import random
    import datetime
    
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    sales_data = {
        'Date': dates,
        'Sales': [random.randint(1000, 5000) + 500 * (i % 30) for i in range(len(dates))],
        'Marketing_Spend': [random.randint(100, 800) for _ in range(len(dates))],
        'Temperature': [random.randint(0, 35) for _ in range(len(dates))],
        'Day_of_Week': [date.strftime('%A') for date in dates],
        'Month': [date.month for date in dates],
        'Holiday': [random.choice(['Yes', 'No']) if random.random() < 0.1 else 'No' for _ in range(len(dates))]
    }
    
    df_sales = pd.DataFrame(sales_data)
    sales_file = data_dir / "sales_data.csv"
    df_sales.head(100).to_csv(sales_file, index=False)  # Save first 100 rows
    print(f"   ✅ Created: {sales_file}")
    
    # Dataset 3: Customer Data (with missing values and data quality issues)
    print("👥 Creating Customer Dataset (with data quality issues)...")
    customer_data = {
        'Customer_ID': [f'CUST_{i:04d}' for i in range(1, 21)],
        'Name': ['John Smith', 'Jane Doe', None, 'Bob Wilson', 'Alice Johnson',
                'Charlie Brown', None, 'Diana Prince', 'Frank Castle', 'Grace Kelly',
                'Henry Ford', 'Isabel Archer', 'Jack Ryan', None, 'Leo Tolstoy',
                'Mia Wallace', 'Noah Webster', 'Olivia Pope', 'Paul McCartney', 'Quinn Fabray'],
        'Age': [25, 35, 28, None, 42, 31, 29, 38, None, 33, 45, 27, 39, 31, 44, 26, 37, 32, 41, 29],
        'Email': ['john@email.com', 'jane@email.com', 'invalid-email', 'bob@email.com', None,
                 'charlie@email.com', 'diana@', 'diana@email.com', 'frank@email.com', None,
                 'henry@email.com', 'isabel@email.com', 'jack@email.com', 'karen@email.com', 'leo@email.com',
                 'mia@email.com', None, 'olivia@email.com', 'paul@email.com', 'quinn@email.com'],
        'Purchase_Amount': [100, 250, 300, 150, None, 200, 175, 400, 225, 300,
                           500, None, 350, 275, 450, 125, 375, 325, None, 180],
        'Category': ['Electronics', 'Clothing', 'Electronics', 'Books', 'Electronics',
                    'Clothing', 'Electronics', 'Books', 'Clothing', 'Electronics',
                    'Books', 'Clothing', 'Electronics', 'Books', 'Electronics',
                    'Clothing', 'Books', 'Electronics', 'Clothing', 'Books'],
        'Satisfaction': [4, 5, 3, 4, 5, 3, 4, 5, 4, 3, 5, 4, 3, 4, 5, 3, 4, 5, 4, 3]
    }
    
    df_customers = pd.DataFrame(customer_data)
    customer_file = data_dir / "customer_data.csv"
    df_customers.to_csv(customer_file, index=False)
    print(f"   ✅ Created: {customer_file}")
    
    print(f"\n✅ All sample datasets created in {data_dir}/")
    return [employee_file, sales_file, customer_file]

def upload_and_analyze_dataset(file_path, dataset_name):
    """Upload and analyze a dataset"""
    print_step("UPLOAD", f"Uploading {dataset_name}")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/api/dataset/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            dataset_id = result['dataset_id']
            
            print("✅ Upload successful!")
            print(f"   📊 Dataset ID: {dataset_id}")
            print(f"   📈 Rows: {result['rows']:,}")
            print(f"   📋 Columns: {result['columns']}")
            print(f"   🎯 Data Quality Score: {result['data_quality_score']}/100")
            
            # Show column information
            print(f"\n📋 Column Analysis:")
            for col in result['column_info']:
                missing_info = f" ({col['missing_percentage']}% missing)" if col['missing_percentage'] > 0 else ""
                print(f"   • {col['name']}: {col['data_type']} ({col['unique_count']} unique){missing_info}")
            
            # Show suggestions
            if result.get('suggestions'):
                print(f"\n💡 Suggestions:")
                for suggestion in result['suggestions']:
                    print(f"   • {suggestion}")
            
            # Show warnings
            if result.get('warnings'):
                print(f"\n⚠️  Warnings:")
                for warning in result['warnings']:
                    print(f"   • {warning}")
            
            return dataset_id, result
        else:
            print(f"❌ Upload failed: {response.text}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        return None, None

def demonstrate_api_endpoints():
    """Demonstrate various API endpoints"""
    print_header("API ENDPOINTS DEMONSTRATION")
    
    # 1. Health Check
    print_step(1, "Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(f"GET /health")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # 2. List Datasets
    print_step(2, "List All Datasets")
    response = requests.get(f"{BASE_URL}/api/datasets")
    print(f"GET /api/datasets")
    result = response.json()
    print(f"Response: Found {len(result['datasets'])} datasets")
    for dataset in result['datasets']:
        print(f"   • {dataset['filename']} ({dataset['rows']} rows, {dataset['columns']} cols)")

def demonstrate_web_interface():
    """Show how to access the web interface"""
    print_header("WEB INTERFACE ACCESS")
    
    print("🌐 You can access NoCodeML through your web browser:")
    print(f"   • Main Interface: {BASE_URL}")
    print(f"   • API Documentation: {BASE_URL}/api/docs")
    print(f"   • Alternative Docs: {BASE_URL}/api/redoc")
    print(f"   • Health Check: {BASE_URL}/health")
    
    print("\n📱 The web interface provides:")
    print("   • Beautiful, user-friendly dashboard")
    print("   • Interactive API documentation")
    print("   • Real-time dataset analysis")
    print("   • Upload functionality via web browser")

def demonstrate_curl_commands():
    """Show equivalent curl commands"""
    print_header("CURL COMMAND EXAMPLES")
    
    print("📋 You can also interact with NoCodeML using curl commands:")
    
    print(f"\n1️⃣  Health Check:")
    print(f"   curl {BASE_URL}/health")
    
    print(f"\n2️⃣  Upload Dataset:")
    print(f"   curl -X POST \"{BASE_URL}/api/dataset/upload\" \\")
    print(f"        -H \"Content-Type: multipart/form-data\" \\")
    print(f"        -F \"file=@data/samples/employee_performance.csv\"")
    
    print(f"\n3️⃣  List Datasets:")
    print(f"   curl {BASE_URL}/api/datasets")
    
    print(f"\n4️⃣  Get Dataset Info:")
    print(f"   curl {BASE_URL}/api/dataset/{{dataset_id}}/info")

def demonstrate_python_client():
    """Show how to use Python requests"""
    print_header("PYTHON CLIENT EXAMPLES")
    
    python_code = '''
import requests

# 1. Upload a dataset
with open('your_dataset.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/api/dataset/upload', files=files)
    result = response.json()
    dataset_id = result['dataset_id']

# 2. Get dataset information
info_response = requests.get(f'http://localhost:8000/api/dataset/{dataset_id}/info')
dataset_info = info_response.json()

# 3. List all datasets
list_response = requests.get('http://localhost:8000/api/datasets')
all_datasets = list_response.json()
'''
    
    print("📋 Python code example:")
    print(python_code)

def run_complete_demo():
    """Run the complete demonstration"""
    print("🚀 Welcome to the NoCodeML Demo!")
    print("This demo will show you all the features of the NoCodeML platform.")
    
    # Check server health first
    if not check_server_health():
        print("\n❌ Cannot continue demo - server is not running")
        print("💡 Please start the server first: python minimal_server.py")
        return
    
    # Create sample datasets
    sample_files = create_sample_datasets()
    
    # Demonstrate uploading and analyzing each dataset
    print_header("DATASET UPLOAD & ANALYSIS DEMO")
    
    datasets = [
        ("Employee Performance Data", sample_files[0]),
        ("Sales Time Series Data", sample_files[1]),
        ("Customer Data (with quality issues)", sample_files[2])
    ]
    
    uploaded_datasets = []
    
    for name, file_path in datasets:
        dataset_id, result = upload_and_analyze_dataset(file_path, name)
        if dataset_id:
            uploaded_datasets.append((dataset_id, name, result))
        time.sleep(1)  # Small delay between uploads
    
    # Show API endpoints
    demonstrate_api_endpoints()
    
    # Show web interface access
    demonstrate_web_interface()
    
    # Show curl examples
    demonstrate_curl_commands()
    
    # Show Python client examples
    demonstrate_python_client()
    
    # Final summary
    print_header("DEMO SUMMARY")
    print("✅ Demo completed successfully!")
    print(f"📊 Uploaded {len(uploaded_datasets)} datasets")
    print("🌐 Server is running and ready for use")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Open {BASE_URL} in your browser")
    print(f"   2. Explore the API docs at {BASE_URL}/api/docs")
    print(f"   3. Upload your own datasets")
    print(f"   4. Build ML models with your data")
    
    print(f"\n🛑 To stop the server:")
    print(f"   pkill -f minimal_server.py")

if __name__ == "__main__":
    run_complete_demo()
