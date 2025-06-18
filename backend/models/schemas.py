"""
Data models and schemas for NoCodeML API

This module defines Pydantic models for request/response data structures
used throughout the NoCodeML API.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime

class ProblemType(str, Enum):
    """ML problem types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"

class DataType(str, Enum):
    """Data column types"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATETIME = "datetime"
    BINARY = "binary"
    IMAGE = "image"

class ModelStatus(str, Enum):
    """Model training status"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"

# Dataset Schemas
class ColumnInfo(BaseModel):
    """Information about a dataset column"""
    name: str
    data_type: DataType
    missing_count: int
    missing_percentage: float
    unique_count: int
    sample_values: List[Any]
    statistics: Optional[Dict[str, Any]] = None

class DatasetInfo(BaseModel):
    """Dataset information and analysis"""
    filename: str
    size_mb: float
    rows: int
    columns: int
    column_info: List[ColumnInfo]
    data_quality_score: float
    recommended_problem_types: List[ProblemType]
    suggestions: List[str]
    warnings: List[str]
    upload_timestamp: datetime

class DatasetUploadResponse(BaseModel):
    """Response after dataset upload"""
    dataset_id: str
    message: str
    dataset_info: DatasetInfo

# Analysis Schemas
class RecommendationConfig(BaseModel):
    """ML pipeline recommendation configuration"""
    problem_type: ProblemType
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    cv_folds: int = Field(default=5, ge=3, le=10)
    max_training_time: int = Field(default=300, ge=60, le=3600)  # seconds

class ModelRecommendation(BaseModel):
    """Recommended model configuration"""
    algorithm: str
    algorithm_type: str  # 'classical_ml', 'deep_learning', 'ensemble'
    confidence_score: float
    expected_performance: Optional[Dict[str, float]] = None
    hyperparameters: Dict[str, Any]
    preprocessing_steps: List[str]
    explanation: str

class AnalysisResult(BaseModel):
    """Complete analysis result"""
    dataset_id: str
    problem_type: ProblemType
    target_column: Optional[str]
    feature_columns: List[str]
    recommended_models: List[ModelRecommendation]
    data_preprocessing_required: bool
    estimated_training_time: int  # minutes
    analysis_timestamp: datetime

# Model Training Schemas
class ModelConfig(BaseModel):
    """Model training configuration"""
    dataset_id: str
    model_name: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    problem_type: ProblemType
    target_column: str
    feature_columns: List[str]
    preprocessing_config: Dict[str, Any]
    validation_config: Dict[str, Any]

class TrainingStatus(BaseModel):
    """Model training status"""
    model_id: str
    status: ModelStatus
    progress_percentage: float
    current_step: str
    elapsed_time: int  # seconds
    estimated_remaining_time: Optional[int] = None
    logs: List[str]

class ModelMetrics(BaseModel):
    """Model evaluation metrics"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None

class TrainedModel(BaseModel):
    """Trained model information"""
    model_id: str
    model_name: str
    algorithm: str
    problem_type: ProblemType
    dataset_id: str
    training_config: ModelConfig
    metrics: ModelMetrics
    feature_importance: Optional[Dict[str, float]] = None
    model_explanation: str
    training_time: int  # seconds
    created_at: datetime
    model_size_mb: float

# Prediction Schemas
class PredictionRequest(BaseModel):
    """Request for model prediction"""
    model_id: str
    input_data: Union[Dict[str, Any], List[Dict[str, Any]]]
    return_probabilities: bool = False
    explain_prediction: bool = False

class PredictionResponse(BaseModel):
    """Model prediction response"""
    model_id: str
    predictions: Union[Any, List[Any]]
    probabilities: Optional[Union[Dict[str, float], List[Dict[str, float]]]] = None
    explanations: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    prediction_timestamp: datetime

# Error Schemas
class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime

# Health and Status
class HealthStatus(BaseModel):
    """API health status"""
    status: str
    service: str
    version: str
    timestamp: datetime
    uptime: int  # seconds

