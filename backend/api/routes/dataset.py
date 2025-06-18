"""
Dataset API routes for NoCodeML

This module handles dataset upload, analysis, and management operations.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import uuid
import os
from datetime import datetime
from pathlib import Path
from typing import List
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from backend.models.schemas import (
    DatasetInfo, DatasetUploadResponse, ColumnInfo, 
    DataType, ProblemType, ErrorResponse
)
from core.analyzers.dataset_analyzer import DatasetAnalyzer

router = APIRouter()

# Configure upload directory
UPLOAD_DIR = project_root / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Supported file formats
SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls', '.json', '.parquet'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload and analyze a dataset
    
    Accepts CSV, Excel, JSON, or Parquet files and performs initial analysis
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
            )
        
        # Check file size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Generate unique dataset ID and save file
        dataset_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{dataset_id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Analyze the dataset
        analyzer = DatasetAnalyzer()
        dataset_info = await analyzer.analyze_dataset(str(file_path))
        
        return DatasetUploadResponse(
            dataset_id=dataset_id,
            message="Dataset uploaded and analyzed successfully",
            dataset_info=dataset_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")

@router.get("/list")
async def list_datasets():
    """List all uploaded datasets"""
    try:
        datasets = []
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                # Extract dataset ID from filename
                filename = file_path.name
                if "_" in filename:
                    dataset_id = filename.split("_")[0]
                    original_name = "_".join(filename.split("_")[1:])
                    
                    # Get file stats
                    stat = file_path.stat()
                    datasets.append({
                        "dataset_id": dataset_id,
                        "filename": original_name,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "upload_date": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        return {"datasets": datasets}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")

@router.get("/{dataset_id}/info", response_model=DatasetInfo)
async def get_dataset_info(dataset_id: str):
    """Get detailed information about a specific dataset"""
    try:
        # Find the dataset file
        dataset_files = list(UPLOAD_DIR.glob(f"{dataset_id}_*"))
        if not dataset_files:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        file_path = dataset_files[0]
        
        # Re-analyze the dataset
        analyzer = DatasetAnalyzer()
        dataset_info = await analyzer.analyze_dataset(str(file_path))
        
        return dataset_info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving dataset info: {str(e)}")

@router.get("/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, rows: int = 10):
    """Get a preview of the dataset (first N rows)"""
    try:
        # Find the dataset file
        dataset_files = list(UPLOAD_DIR.glob(f"{dataset_id}_*"))
        if not dataset_files:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        file_path = dataset_files[0]
        
        # Load and preview the data
        analyzer = DatasetAnalyzer()
        df = analyzer._load_dataset(str(file_path))
        
        preview_data = df.head(rows).to_dict('records')
        
        return {
            "dataset_id": dataset_id,
            "total_rows": len(df),
            "preview_rows": len(preview_data),
            "columns": list(df.columns),
            "data": preview_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error previewing dataset: {str(e)}")

@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    try:
        # Find and delete the dataset file
        dataset_files = list(UPLOAD_DIR.glob(f"{dataset_id}_*"))
        if not dataset_files:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        for file_path in dataset_files:
            file_path.unlink()
        
        return {"message": f"Dataset {dataset_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting dataset: {str(e)}")

@router.get("/{dataset_id}/statistics")
async def get_dataset_statistics(dataset_id: str):
    """Get detailed statistics for the dataset"""
    try:
        # Find the dataset file
        dataset_files = list(UPLOAD_DIR.glob(f"{dataset_id}_*"))
        if not dataset_files:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        file_path = dataset_files[0]
        
        # Generate detailed statistics
        analyzer = DatasetAnalyzer()
        df = analyzer._load_dataset(str(file_path))
        
        stats = {
            "basic_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "duplicated_rows": df.duplicated().sum()
            },
            "column_statistics": {},
            "correlations": {},
            "missing_data_heatmap": {}
        }
        
        # Column-wise statistics
        for column in df.columns:
            col_stats = {
                "dtype": str(df[column].dtype),
                "non_null_count": df[column].count(),
                "null_count": df[column].isnull().sum(),
                "unique_count": df[column].nunique()
            }
            
            if df[column].dtype in ['int64', 'float64']:
                col_stats.update({
                    "mean": df[column].mean(),
                    "std": df[column].std(),
                    "min": df[column].min(),
                    "max": df[column].max(),
                    "median": df[column].median(),
                    "quartiles": {
                        "q1": df[column].quantile(0.25),
                        "q3": df[column].quantile(0.75)
                    }
                })
            
            stats["column_statistics"][column] = col_stats
        
        # Correlation matrix for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            stats["correlations"] = corr_matrix.to_dict()
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating statistics: {str(e)}")

