"""
Analysis API routes for NoCodeML

This module handles ML pipeline analysis and recommendations.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from backend.models.schemas import (
    RecommendationConfig, AnalysisResult, ModelRecommendation,
    ProblemType, DataType
)
from core.analyzers.recommendation_engine import RecommendationEngine

router = APIRouter()

@router.post("/recommend/{dataset_id}", response_model=AnalysisResult)
async def get_recommendations(dataset_id: str, config: RecommendationConfig):
    """
    Get intelligent ML model recommendations based on dataset analysis and problem type
    """
    try:
        # Use real recommendation engine with intelligent analysis
        engine = RecommendationEngine()
        result = await engine.get_recommendations(dataset_id, config)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@router.get("/algorithms")
async def list_available_algorithms():
    """List all available ML algorithms by category"""
    try:
        algorithms = {
            "classification": {
                "classical_ml": [
                    "Logistic Regression",
                    "Random Forest",
                    "SVM",
                    "Naive Bayes",
                    "Decision Tree",
                    "K-Nearest Neighbors"
                ],
                "ensemble": [
                    "XGBoost",
                    "LightGBM",
                    "AdaBoost",
                    "Gradient Boosting"
                ],
                "deep_learning": [
                    "Neural Network",
                    "CNN (for image data)",
                    "RNN (for sequence data)"
                ]
            },
            "regression": {
                "classical_ml": [
                    "Linear Regression",
                    "Ridge Regression",
                    "Lasso Regression",
                    "Random Forest Regressor",
                    "SVR",
                    "Decision Tree Regressor"
                ],
                "ensemble": [
                    "XGBoost Regressor",
                    "LightGBM Regressor",
                    "Gradient Boosting Regressor"
                ],
                "deep_learning": [
                    "Neural Network Regressor",
                    "CNN Regressor",
                    "RNN Regressor"
                ]
            },
            "clustering": {
                "classical_ml": [
                    "K-Means",
                    "DBSCAN",
                    "Hierarchical Clustering",
                    "Gaussian Mixture Models"
                ]
            },
            "anomaly_detection": {
                "classical_ml": [
                    "Isolation Forest",
                    "One-Class SVM",
                    "Local Outlier Factor"
                ]
            }
        }
        
        return {"algorithms": algorithms}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing algorithms: {str(e)}")

