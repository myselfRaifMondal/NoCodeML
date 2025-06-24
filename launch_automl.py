#!/usr/bin/env python3
"""
AutoML Pipeline Launcher
========================

A simple launcher script that sets up and runs the automated ML pipeline
with different interfaces (CLI or Web).

Usage:
    python launch_automl.py --mode web    # Launch web interface
    python launch_automl.py --mode cli    # Use command line interface
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import pkg_resources
import importlib.util

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'xgboost', 'matplotlib',
        'seaborn', 'plotly', 'optuna', 'streamlit', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            pkg_resources.get_distribution(package)
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n🔧 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def install_dependencies():
    """Install missing dependencies"""
    print("🔧 Installing dependencies...")
    
    # Get the project root directory
    project_root = Path(__file__).parent
    requirements_file = project_root / "requirements.txt"
    
    if requirements_file.exists():
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)])
            print("✅ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    else:
        # Install essential packages manually
        essential_packages = [
            'pandas>=2.1.0',
            'numpy>=1.26.0',
            'scikit-learn>=1.4.0',
            'xgboost>=2.0.0',
            'matplotlib>=3.9.0',
            'seaborn==0.13.0',
            'plotly==5.17.0',
            'optuna>=3.5.0',
            'streamlit>=1.28.0',
            'tqdm==4.66.1'
        ]
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + essential_packages)
            print("✅ Essential packages installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install packages: {e}")
            return False

def launch_web_interface():
    """Launch the Streamlit web interface"""
    print("🚀 Launching AutoML Web Interface...")
    
    # Get the path to the web app
    web_app_path = Path(__file__).parent / "automated_ml_web_app.py"
    
    if not web_app_path.exists():
        print(f"❌ Web app file not found: {web_app_path}")
        return False
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            str(web_app_path),
            '--server.port', '8501',
            '--server.address', 'localhost'
        ])
        return True
    except KeyboardInterrupt:
        print("\n👋 AutoML Web Interface stopped.")
        return True
    except Exception as e:
        print(f"❌ Failed to launch web interface: {e}")
        return False

def launch_cli_interface():
    """Show CLI usage instructions"""
    print("🖥️  AutoML Command Line Interface")
    print("=" * 50)
    
    cli_script_path = Path(__file__).parent / "automated_ml_pipeline.py"
    
    if not cli_script_path.exists():
        print(f"❌ CLI script not found: {cli_script_path}")
        return False
    
    print("Usage:")
    print(f"python {cli_script_path} --input <data_file> --target <target_column> [--output <output_dir>]")
    print()
    print("Examples:")
    print(f"python {cli_script_path} --input data.csv --target price")
    print(f"python {cli_script_path} --input https://example.com/data.csv --target category")
    print(f"python {cli_script_path} --input data.xlsx --target target --output my_results")
    print()
    print("Options:")
    print("  --input    : Input data file path or URL (CSV, Excel, JSON)")
    print("  --target   : Target column name to predict")
    print("  --output   : Output directory for results (default: automl_output)")
    print()
    print("The pipeline will automatically:")
    print("  📊 Analyze your data")
    print("  🧹 Clean and prepare it")
    print("  🎯 Select best features")
    print("  🤖 Train multiple models")
    print("  📦 Export the best model as .pkl file")
    
    return True

def create_sample_data():
    """Create a sample dataset for testing"""
    print("📋 Creating sample dataset...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample classification dataset
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.normal(50000, 15000, n_samples),
            'education_years': np.random.randint(8, 20, n_samples),
            'experience': np.random.randint(0, 40, n_samples),
            'city_size': np.random.choice(['Small', 'Medium', 'Large'], n_samples),
            'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], n_samples),
            'satisfaction_score': np.random.uniform(1, 10, n_samples)
        }
        
        # Create target variable (binary classification)
        df = pd.DataFrame(data)
        # High performer based on some criteria
        df['high_performer'] = (
            (df['satisfaction_score'] > 7) & 
            (df['experience'] > 5) & 
            (df['income'] > 45000)
        ).astype(int)
        
        # Add some missing values
        df.loc[np.random.choice(df.index, 50), 'income'] = np.nan
        df.loc[np.random.choice(df.index, 30), 'satisfaction_score'] = np.nan
        
        # Save sample data
        sample_file = Path(__file__).parent / "sample_employee_data.csv"
        df.to_csv(sample_file, index=False)
        
        print(f"✅ Sample dataset created: {sample_file}")
        print(f"   📊 Shape: {df.shape}")
        print(f"   🎯 Target: high_performer (binary classification)")
        print(f"   📋 Features: {', '.join(df.columns[:-1])}")
        
        return str(sample_file)
        
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return None

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="AutoML Pipeline Launcher")
    parser.add_argument('--mode', choices=['web', 'cli', 'setup', 'sample'], 
                       default='web', help='Launch mode')
    parser.add_argument('--install-deps', action='store_true', 
                       help='Install missing dependencies')
    
    args = parser.parse_args()
    
    print("🤖 AutoML Pipeline Launcher")
    print("=" * 40)
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_dependencies():
            sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\n🔧 Run with --install-deps to install missing packages:")
        print(f"python {__file__} --install-deps")
        sys.exit(1)
    
    # Handle different modes
    if args.mode == 'web':
        if not launch_web_interface():
            sys.exit(1)
    
    elif args.mode == 'cli':
        if not launch_cli_interface():
            sys.exit(1)
    
    elif args.mode == 'sample':
        sample_file = create_sample_data()
        if sample_file:
            print(f"\n🚀 Test the pipeline with your sample data:")
            print(f"python {Path(__file__).parent / 'automated_ml_pipeline.py'} --input {sample_file} --target high_performer")
    
    elif args.mode == 'setup':
        print("🔧 Setup Mode")
        print("-" * 20)
        
        # Check Python version
        python_version = sys.version_info
        print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version < (3, 8):
            print("⚠️  Warning: Python 3.8+ recommended")
        else:
            print("✅ Python version OK")
        
        # Check dependencies
        check_dependencies()
        
        # Create sample data
        create_sample_data()
        
        print("\n🎉 Setup complete! Choose your interface:")
        print(f"  Web Interface: python {__file__} --mode web")
        print(f"  CLI Interface: python {__file__} --mode cli")

if __name__ == "__main__":
    main()
