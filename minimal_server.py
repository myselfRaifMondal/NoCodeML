#!/usr/bin/env python3
"""
Minimal NoCodeML Server

A simplified version of the NoCodeML server that runs without the problematic modules.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
import uuid
import os

# Initialize FastAPI app
app = FastAPI(
    title="NoCodeML API (Minimal)",
    description="No-Code Machine Learning Platform API - Minimal Version",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store for uploaded datasets
datasets_store = {}

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - serves the frontend or API info"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NoCodeML</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                   margin: 0; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; min-height: 100vh; }
            .container { max-width: 800px; margin: 0 auto; text-align: center; }
            h1 { font-size: 3em; margin-bottom: 20px; }
            p { font-size: 1.2em; margin-bottom: 30px; opacity: 0.9; }
            .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                       gap: 20px; margin: 40px 0; }
            .feature { background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; }
            a { color: #ffd700; text-decoration: none; font-weight: bold; }
            a:hover { text-decoration: underline; }
            .status { background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 NoCodeML</h1>
            <p>No-Code Machine Learning Platform</p>
            <p>Democratizing AI/ML model building for everyone!</p>
            
            <div class="status">
                <h3>✅ Server Running Successfully!</h3>
                <p>The NoCodeML server is now running and ready to receive requests.</p>
            </div>
            
            <div class="features">
                <div class="feature">
                    <h3>🔍 Dataset Analysis</h3>
                    <p>Upload and analyze your datasets through the API</p>
                </div>
                <div class="feature">
                    <h3>📊 Data Profiling</h3>
                    <p>Get detailed insights about your data quality</p>
                </div>
                <div class="feature">
                    <h3>🚀 Ready to Use</h3>
                    <p>API endpoints are ready for integration</p>
                </div>
            </div>
            
            <p>
                <a href="/api/docs">📚 API Documentation</a> | 
                <a href="/api/redoc">📖 ReDoc</a> | 
                <a href="/health">💚 Health Check</a>
            </p>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "NoCodeML API (Minimal)", "version": "1.0.0"}

@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and analyze a dataset"""
    try:
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
        
        # Generate dataset ID
        dataset_id = str(uuid.uuid4())
        
        # Save file
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / f"{dataset_id}_{file.filename}"
        
        # Read and save file content
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Load and analyze dataset
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Basic analysis
        analysis = analyze_dataset_basic(df, file.filename)
        analysis["dataset_id"] = dataset_id
        
        # Store dataset info
        datasets_store[dataset_id] = {
            "file_path": str(file_path),
            "filename": file.filename,
            "analysis": analysis,
            "upload_timestamp": datetime.now().isoformat()
        }
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")

@app.get("/api/dataset/{dataset_id}/info")
async def get_dataset_info(dataset_id: str):
    """Get information about an uploaded dataset"""
    if dataset_id not in datasets_store:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return datasets_store[dataset_id]["analysis"]

@app.get("/api/datasets")
async def list_datasets():
    """List all uploaded datasets"""
    return {
        "datasets": [
            {
                "dataset_id": dataset_id,
                "filename": info["filename"],
                "upload_timestamp": info["upload_timestamp"],
                "rows": info["analysis"]["rows"],
                "columns": info["analysis"]["columns"]
            }
            for dataset_id, info in datasets_store.items()
        ]
    }

def analyze_dataset_basic(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
    """Perform basic dataset analysis"""
    try:
        # Basic statistics
        rows, columns = df.shape
        missing_values = df.isnull().sum().sum()
        missing_percentage = (missing_values / (rows * columns)) * 100
        
        # Column analysis
        column_info = []
        for col in df.columns:
            col_data = df[col]
            missing_count = col_data.isnull().sum()
            unique_count = col_data.nunique()
            
            # Determine data type
            if pd.api.types.is_numeric_dtype(col_data):
                data_type = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                data_type = "datetime"
            else:
                data_type = "categorical"
            
            column_info.append({
                "name": col,
                "data_type": data_type,
                "missing_count": int(missing_count),
                "missing_percentage": round((missing_count / rows) * 100, 2),
                "unique_count": int(unique_count),
                "sample_values": col_data.dropna().head(3).tolist()
            })
        
        # Generate suggestions
        suggestions = []
        if rows < 100:
            suggestions.append("Consider collecting more data for better model performance")
        if missing_percentage > 10:
            suggestions.append("High missing values detected - consider data cleaning")
        
        # Generate warnings
        warnings = []
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f"{duplicate_count} duplicate rows detected")
        
        # Calculate quality score
        quality_score = max(0, 100 - missing_percentage - (duplicate_count / rows * 10))
        
        return {
            "filename": filename,
            "rows": rows,
            "columns": columns,
            "column_info": column_info,
            "data_quality_score": round(quality_score, 1),
            "missing_values_total": int(missing_values),
            "missing_percentage": round(missing_percentage, 2),
            "duplicate_rows": int(duplicate_count),
            "suggested_problem_types": ["classification", "regression"],
            "suggestions": suggestions,
            "warnings": warnings,
            "upload_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "filename": filename,
            "error": f"Analysis failed: {str(e)}",
            "upload_timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    uvicorn.run(
        "minimal_server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
