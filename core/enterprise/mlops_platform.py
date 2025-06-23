"""
Enterprise MLOps Platform for NoCodeML

This module provides enterprise-grade ML operations including:
- Automated ML lifecycle management
- Model versioning and registry
- A/B testing and deployment strategies
- Performance monitoring and drift detection
- Automated retraining pipelines
- Compliance and governance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
import uuid
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import hashlib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.models.schemas import ProblemType

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    TRAINING = "training"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"

class DeploymentStrategy(Enum):
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    SHADOW = "shadow"

class DriftType(Enum):
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DRIFT = "performance_drift"

@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    model_id: str
    name: str
    version: str
    algorithm: str
    problem_type: str
    status: ModelStatus
    created_at: datetime
    created_by: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    features: List[str]
    training_data_hash: str
    model_size_mb: float
    inference_time_ms: float
    tags: List[str]
    description: str
    business_impact: Dict[str, Any]
    compliance_info: Dict[str, Any]

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    strategy: DeploymentStrategy
    traffic_percentage: float
    rollback_threshold: float
    monitoring_window_hours: int
    auto_promote: bool
    approval_required: bool
    health_check_endpoint: str

@dataclass
class MonitoringAlert:
    """Monitoring alert definition"""
    alert_id: str
    model_id: str
    alert_type: str
    severity: str
    threshold: float
    current_value: float
    triggered_at: datetime
    message: str
    resolved: bool

class EnterpriseMLOps:
    """Enterprise-grade MLOps platform"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.model_registry_path = project_root / "models" / "registry"
        self.experiment_path = project_root / "experiments"
        self.monitoring_path = project_root / "monitoring"
        
        # Initialize directories
        for path in [self.model_registry_path, self.experiment_path, self.monitoring_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(str(self.experiment_path / "mlruns"))
        
        # Initialize database
        self.db_path = self.monitoring_path / "mlops.db"
        self._init_database()
        
        # Active deployments
        self.active_deployments = {}
        self.monitoring_jobs = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load MLOps configuration"""
        default_config = {
            "monitoring": {
                "drift_threshold": 0.1,
                "performance_threshold": 0.05,
                "check_interval_minutes": 60,
                "alert_channels": ["email", "slack"]
            },
            "deployment": {
                "default_strategy": "canary",
                "default_traffic_split": 0.1,
                "auto_rollback": True,
                "approval_required": False
            },
            "model_governance": {
                "retention_days": 90,
                "require_documentation": True,
                "require_bias_testing": True,
                "require_explainability": True
            },
            "compliance": {
                "data_lineage_tracking": True,
                "audit_logging": True,
                "encryption_at_rest": True,
                "access_control": True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def _init_database(self):
        """Initialize SQLite database for metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                algorithm TEXT,
                problem_type TEXT,
                status TEXT,
                created_at TIMESTAMP,
                created_by TEXT,
                metrics TEXT,
                hyperparameters TEXT,
                features TEXT,
                training_data_hash TEXT,
                model_size_mb REAL,
                inference_time_ms REAL,
                tags TEXT,
                description TEXT,
                business_impact TEXT,
                compliance_info TEXT
            )
        """)
        
        # Deployments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS deployments (
                deployment_id TEXT PRIMARY KEY,
                model_id TEXT,
                environment TEXT,
                strategy TEXT,
                traffic_percentage REAL,
                deployed_at TIMESTAMP,
                status TEXT,
                config TEXT,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        """)
        
        # Monitoring table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS monitoring_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                timestamp TIMESTAMP,
                metric_name TEXT,
                metric_value REAL,
                environment TEXT,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        """)
        
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                model_id TEXT,
                alert_type TEXT,
                severity TEXT,
                threshold REAL,
                current_value REAL,
                triggered_at TIMESTAMP,
                message TEXT,
                resolved BOOLEAN,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        """)
        
        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                created_at TIMESTAMP,
                created_by TEXT,
                status TEXT,
                config TEXT,
                results TEXT
            )
        """)
        
        # Data lineage table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_lineage (
                lineage_id TEXT PRIMARY KEY,
                entity_id TEXT,
                entity_type TEXT,
                parent_id TEXT,
                relationship_type TEXT,
                created_at TIMESTAMP,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def register_model(self, model_obj: Any, metadata: ModelMetadata, 
                           artifacts: Optional[Dict[str, Any]] = None) -> str:
        """Register a model in the enterprise registry"""
        try:
            # Generate unique model ID if not provided
            if not metadata.model_id:
                metadata.model_id = f"model_{uuid.uuid4().hex[:8]}"
            
            # Create model directory
            model_dir = self.model_registry_path / metadata.model_id
            model_dir.mkdir(exist_ok=True)
            
            # Save model artifacts
            model_path = model_dir / "model.pkl"
            joblib.dump(model_obj, model_path)
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                metadata_dict = asdict(metadata)
                metadata_dict['created_at'] = metadata.created_at.isoformat()
                metadata_dict['status'] = metadata.status.value
                json.dump(metadata_dict, f, indent=2)
            
            # Save additional artifacts
            if artifacts:
                artifacts_dir = model_dir / "artifacts"
                artifacts_dir.mkdir(exist_ok=True)
                
                for artifact_name, artifact_data in artifacts.items():
                    artifact_path = artifacts_dir / f"{artifact_name}.pkl"
                    joblib.dump(artifact_data, artifact_path)
            
            # Log to MLflow
            with mlflow.start_run():
                mlflow.log_params(metadata.hyperparameters)
                mlflow.log_metrics(metadata.metrics)
                mlflow.sklearn.log_model(model_obj, "model")
                mlflow.set_tag("model_id", metadata.model_id)
                mlflow.set_tag("algorithm", metadata.algorithm)
                mlflow.set_tag("problem_type", metadata.problem_type)
            
            # Store in database
            await self._store_model_metadata(metadata)
            
            # Track data lineage
            await self._track_data_lineage(metadata.model_id, "model", 
                                         metadata.training_data_hash, "trained_on")
            
            logger.info(f"Model {metadata.model_id} registered successfully")
            return metadata.model_id
            
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            raise
    
    async def deploy_model(self, model_id: str, environment: str,
                          config: DeploymentConfig) -> str:
        """Deploy model with specified strategy"""
        try:
            deployment_id = f"deploy_{uuid.uuid4().hex[:8]}"
            
            # Validate model exists
            model_metadata = await self.get_model_metadata(model_id)
            if not model_metadata:
                raise ValueError(f"Model {model_id} not found")
            
            # Load model
            model_path = self.model_registry_path / model_id / "model.pkl"
            model = joblib.load(model_path)
            
            # Validate deployment readiness
            await self._validate_deployment_readiness(model_metadata, config)
            
            # Execute deployment strategy
            if config.strategy == DeploymentStrategy.CANARY:
                await self._deploy_canary(model_id, model, environment, config, deployment_id)
            elif config.strategy == DeploymentStrategy.BLUE_GREEN:
                await self._deploy_blue_green(model_id, model, environment, config, deployment_id)
            elif config.strategy == DeploymentStrategy.ROLLING:
                await self._deploy_rolling(model_id, model, environment, config, deployment_id)
            elif config.strategy == DeploymentStrategy.SHADOW:
                await self._deploy_shadow(model_id, model, environment, config, deployment_id)
            
            # Store deployment metadata
            await self._store_deployment_metadata(deployment_id, model_id, environment, config)
            
            # Start monitoring
            await self._start_deployment_monitoring(deployment_id, model_id, environment)
            
            logger.info(f"Model {model_id} deployed with ID {deployment_id}")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {str(e)}")
            raise
    
    async def monitor_model_performance(self, model_id: str, 
                                      prediction_data: pd.DataFrame,
                                      actual_data: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Monitor model performance and detect drift"""
        try:
            monitoring_results = {
                'model_id': model_id,
                'timestamp': datetime.now(),
                'data_drift': False,
                'concept_drift': False,
                'performance_drift': False,
                'alerts': []
            }
            
            # Get model metadata
            model_metadata = await self.get_model_metadata(model_id)
            if not model_metadata:
                raise ValueError(f"Model {model_id} not found")
            
            # Load reference data for comparison
            reference_data = await self._load_reference_data(model_id)
            
            # Data drift detection
            data_drift_score = await self._detect_data_drift(reference_data, prediction_data)
            monitoring_results['data_drift_score'] = data_drift_score
            
            if data_drift_score > self.config['monitoring']['drift_threshold']:
                monitoring_results['data_drift'] = True
                alert = await self._create_alert(model_id, "data_drift", "medium", 
                                               data_drift_score, 
                                               "Significant data drift detected")
                monitoring_results['alerts'].append(alert)
            
            # Performance drift detection (if actual values provided)
            if actual_data is not None:
                model = await self._load_model(model_id)
                predictions = model.predict(prediction_data)
                
                current_performance = await self._calculate_performance_metrics(
                    actual_data, predictions, model_metadata['problem_type']
                )
                
                reference_performance = model_metadata['metrics']
                performance_drift = await self._detect_performance_drift(
                    reference_performance, current_performance
                )
                
                monitoring_results['performance_drift'] = performance_drift
                monitoring_results['current_performance'] = current_performance
                
                if performance_drift:
                    alert = await self._create_alert(model_id, "performance_drift", "high",
                                                   0.0, "Model performance degradation detected")
                    monitoring_results['alerts'].append(alert)
            
            # Store monitoring data
            await self._store_monitoring_data(model_id, monitoring_results)
            
            # Trigger alerts if necessary
            for alert in monitoring_results['alerts']:
                await self._trigger_alert(alert)
            
            return monitoring_results
            
        except Exception as e:
            logger.error(f"Failed to monitor model performance: {str(e)}")
            raise
    
    async def auto_retrain_model(self, model_id: str, new_data: pd.DataFrame,
                               target_column: str) -> Optional[str]:
        """Automatically retrain model when drift is detected"""
        try:
            # Get original model metadata
            original_metadata = await self.get_model_metadata(model_id)
            if not original_metadata:
                raise ValueError(f"Model {model_id} not found")
            
            # Load original model and preprocessing pipeline
            model_path = self.model_registry_path / model_id / "model.pkl"
            preprocessing_path = self.model_registry_path / model_id / "artifacts" / "preprocessor.pkl"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found for {model_id}")
            
            # Prepare new training data
            if preprocessing_path.exists():
                preprocessor = joblib.load(preprocessing_path)
                X_new = preprocessor.transform(new_data.drop(columns=[target_column]))
            else:
                X_new = new_data.drop(columns=[target_column])
            
            y_new = new_data[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_new, y_new, test_size=0.2, random_state=42
            )
            
            # Load and retrain model
            from sklearn.base import clone
            original_model = joblib.load(model_path)
            new_model = clone(original_model)
            new_model.fit(X_train, y_train)
            
            # Evaluate new model
            predictions = new_model.predict(X_test)
            new_metrics = await self._calculate_performance_metrics(
                y_test, predictions, original_metadata['problem_type']
            )
            
            # Compare with original performance
            original_metrics = original_metadata['metrics']
            primary_metric = 'accuracy' if original_metadata['problem_type'] == 'classification' else 'r2_score'
            
            if new_metrics.get(primary_metric, 0) > original_metrics.get(primary_metric, 0):
                # New model is better, register it
                new_metadata = ModelMetadata(
                    model_id=f"{model_id}_retrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    name=f"{original_metadata['name']}_retrained",
                    version=f"{original_metadata['version']}.1",
                    algorithm=original_metadata['algorithm'],
                    problem_type=original_metadata['problem_type'],
                    status=ModelStatus.STAGING,
                    created_at=datetime.now(),
                    created_by="auto_retrain_system",
                    metrics=new_metrics,
                    hyperparameters=original_metadata['hyperparameters'],
                    features=original_metadata['features'],
                    training_data_hash=self._calculate_data_hash(new_data),
                    model_size_mb=0.0,  # Will be calculated during registration
                    inference_time_ms=original_metadata['inference_time_ms'],
                    tags=original_metadata['tags'] + ['auto_retrained'],
                    description=f"Auto-retrained version of {original_metadata['name']}",
                    business_impact=original_metadata['business_impact'],
                    compliance_info=original_metadata['compliance_info']
                )
                
                # Register retrained model
                new_model_id = await self.register_model(
                    new_model, new_metadata,
                    artifacts={'preprocessor': preprocessor} if preprocessing_path.exists() else None
                )
                
                logger.info(f"Successfully retrained model. New model ID: {new_model_id}")
                return new_model_id
            else:
                logger.info("Retrained model did not improve performance. Keeping original model.")
                return None
                
        except Exception as e:
            logger.error(f"Failed to auto-retrain model: {str(e)}")
            raise
    
    async def get_model_explainability(self, model_id: str, 
                                     sample_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate model explainability insights"""
        try:
            # Load model and metadata
            model = await self._load_model(model_id)
            metadata = await self.get_model_metadata(model_id)
            
            explainability_results = {
                'model_id': model_id,
                'global_importance': {},
                'local_explanations': [],
                'fairness_metrics': {},
                'bias_analysis': {}
            }
            
            # Global feature importance
            if hasattr(model, 'feature_importances_'):
                feature_names = metadata['features']
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                explainability_results['global_importance'] = importance_dict
            elif hasattr(model, 'coef_'):
                feature_names = metadata['features']
                importance_dict = dict(zip(feature_names, np.abs(model.coef_.flatten())))
                explainability_results['global_importance'] = importance_dict
            
            # SHAP values for local explanations (if available)
            try:
                import shap
                explainer = shap.Explainer(model)
                shap_values = explainer(sample_data.head(100))
                
                explainability_results['local_explanations'] = {
                    'shap_values': shap_values.values.tolist(),
                    'base_values': shap_values.base_values.tolist() if hasattr(shap_values, 'base_values') else [],
                    'feature_names': sample_data.columns.tolist()
                }
            except ImportError:
                logger.warning("SHAP not available for local explanations")
            except Exception as e:
                logger.warning(f"Failed to generate SHAP explanations: {str(e)}")
            
            # Bias analysis for protected attributes
            protected_attributes = ['gender', 'race', 'age_group', 'ethnicity']
            available_protected = [attr for attr in protected_attributes if attr in sample_data.columns]
            
            if available_protected:
                explainability_results['bias_analysis'] = await self._analyze_bias(
                    model, sample_data, available_protected
                )
            
            return explainability_results
            
        except Exception as e:
            logger.error(f"Failed to generate explainability: {str(e)}")
            raise
    
    async def generate_compliance_report(self, model_id: str) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        try:
            metadata = await self.get_model_metadata(model_id)
            if not metadata:
                raise ValueError(f"Model {model_id} not found")
            
            compliance_report = {
                'model_id': model_id,
                'report_generated_at': datetime.now().isoformat(),
                'compliance_status': 'COMPLIANT',
                'requirements_met': {},
                'recommendations': [],
                'audit_trail': await self._get_audit_trail(model_id),
                'data_lineage': await self._get_data_lineage(model_id),
                'risk_assessment': {}
            }
            
            # Check documentation requirements
            compliance_report['requirements_met']['documentation'] = bool(metadata.get('description'))
            
            # Check bias testing
            explainability = await self.get_model_explainability(model_id, pd.DataFrame())
            compliance_report['requirements_met']['bias_testing'] = bool(explainability.get('bias_analysis'))
            
            # Check model governance
            compliance_report['requirements_met']['version_control'] = bool(metadata.get('version'))
            compliance_report['requirements_met']['performance_monitoring'] = await self._check_monitoring_setup(model_id)
            
            # Risk assessment
            compliance_report['risk_assessment'] = {
                'data_privacy_risk': 'LOW',  # Would implement actual risk calculation
                'algorithmic_bias_risk': 'MEDIUM',
                'model_drift_risk': 'LOW',
                'security_risk': 'LOW'
            }
            
            # Generate recommendations
            if not compliance_report['requirements_met']['documentation']:
                compliance_report['recommendations'].append("Add comprehensive model documentation")
            
            if not compliance_report['requirements_met']['bias_testing']:
                compliance_report['recommendations'].append("Implement bias testing for protected attributes")
            
            # Overall compliance status
            all_requirements_met = all(compliance_report['requirements_met'].values())
            compliance_report['compliance_status'] = 'COMPLIANT' if all_requirements_met else 'NON_COMPLIANT'
            
            return compliance_report
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {str(e)}")
            raise
    
    async def run_ab_test(self, model_a_id: str, model_b_id: str,
                         test_data: pd.DataFrame, target_column: str,
                         test_duration_hours: int = 24) -> Dict[str, Any]:
        """Run A/B test between two models"""
        try:
            test_id = f"abtest_{uuid.uuid4().hex[:8]}"
            
            # Load both models
            model_a = await self._load_model(model_a_id)
            model_b = await self._load_model(model_b_id)
            
            # Split test data randomly
            test_data_a = test_data.sample(frac=0.5, random_state=42)
            test_data_b = test_data.drop(test_data_a.index)
            
            # Get predictions
            X_a = test_data_a.drop(columns=[target_column])
            y_a = test_data_a[target_column]
            predictions_a = model_a.predict(X_a)
            
            X_b = test_data_b.drop(columns=[target_column])
            y_b = test_data_b[target_column]
            predictions_b = model_b.predict(X_b)
            
            # Calculate metrics
            metadata_a = await self.get_model_metadata(model_a_id)
            metadata_b = await self.get_model_metadata(model_b_id)
            
            metrics_a = await self._calculate_performance_metrics(
                y_a, predictions_a, metadata_a['problem_type']
            )
            metrics_b = await self._calculate_performance_metrics(
                y_b, predictions_b, metadata_b['problem_type']
            )
            
            # Statistical significance testing
            significance_results = await self._calculate_statistical_significance(
                metrics_a, metrics_b, len(test_data_a), len(test_data_b)
            )
            
            ab_test_results = {
                'test_id': test_id,
                'model_a_id': model_a_id,
                'model_b_id': model_b_id,
                'test_started_at': datetime.now().isoformat(),
                'test_duration_hours': test_duration_hours,
                'model_a_metrics': metrics_a,
                'model_b_metrics': metrics_b,
                'statistical_significance': significance_results,
                'winner': None,
                'confidence_level': 0.95,
                'sample_sizes': {'model_a': len(test_data_a), 'model_b': len(test_data_b)}
            }
            
            # Determine winner
            primary_metric = 'accuracy' if metadata_a['problem_type'] == 'classification' else 'r2_score'
            
            if significance_results.get('significant', False):
                if metrics_a.get(primary_metric, 0) > metrics_b.get(primary_metric, 0):
                    ab_test_results['winner'] = model_a_id
                else:
                    ab_test_results['winner'] = model_b_id
            else:
                ab_test_results['winner'] = 'no_significant_difference'
            
            # Store test results
            await self._store_ab_test_results(ab_test_results)
            
            return ab_test_results
            
        except Exception as e:
            logger.error(f"Failed to run A/B test: {str(e)}")
            raise
    
    # Helper methods
    async def _store_model_metadata(self, metadata: ModelMetadata):
        """Store model metadata in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO models VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            metadata.model_id, metadata.name, metadata.version, metadata.algorithm,
            metadata.problem_type, metadata.status.value, metadata.created_at.isoformat(),
            metadata.created_by, json.dumps(metadata.metrics), 
            json.dumps(metadata.hyperparameters), json.dumps(metadata.features),
            metadata.training_data_hash, metadata.model_size_mb, metadata.inference_time_ms,
            json.dumps(metadata.tags), metadata.description, 
            json.dumps(metadata.business_impact), json.dumps(metadata.compliance_info)
        ))
        
        conn.commit()
        conn.close()
    
    async def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve model metadata from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            metadata = dict(zip(columns, row))
            
            # Parse JSON fields
            for field in ['metrics', 'hyperparameters', 'features', 'tags', 'business_impact', 'compliance_info']:
                if metadata[field]:
                    metadata[field] = json.loads(metadata[field])
            
            return metadata
        return None
    
    async def _load_model(self, model_id: str):
        """Load model from registry"""
        model_path = self.model_registry_path / model_id / "model.pkl"
        if model_path.exists():
            return joblib.load(model_path)
        raise FileNotFoundError(f"Model {model_id} not found")
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of dataset"""
        return hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()
    
    async def _detect_data_drift(self, reference_data: pd.DataFrame, 
                               current_data: pd.DataFrame) -> float:
        """Detect data drift using statistical methods"""
        try:
            from scipy.stats import ks_2samp
            
            drift_scores = []
            common_columns = set(reference_data.columns) & set(current_data.columns)
            
            for col in common_columns:
                if reference_data[col].dtype in ['int64', 'float64']:
                    # KS test for numerical features
                    statistic, p_value = ks_2samp(reference_data[col].dropna(), 
                                                 current_data[col].dropna())
                    drift_scores.append(statistic)
                else:
                    # Chi-square test for categorical features
                    try:
                        ref_counts = reference_data[col].value_counts()
                        curr_counts = current_data[col].value_counts()
                        
                        # Align categories
                        all_categories = set(ref_counts.index) | set(curr_counts.index)
                        ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                        curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
                        
                        from scipy.stats import chisquare
                        statistic, p_value = chisquare(curr_aligned, ref_aligned)
                        drift_scores.append(min(statistic / 100, 1.0))  # Normalize
                    except:
                        drift_scores.append(0.0)
            
            return np.mean(drift_scores) if drift_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate data drift: {str(e)}")
            return 0.0
    
    async def _calculate_performance_metrics(self, y_true, y_pred, problem_type: str) -> Dict[str, float]:
        """Calculate performance metrics based on problem type"""
        metrics = {}
        
        if problem_type.lower() == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        else:  # regression
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2_score'] = r2_score(y_true, y_pred)
        
        return metrics
    
    async def _create_alert(self, model_id: str, alert_type: str, severity: str,
                          current_value: float, message: str) -> MonitoringAlert:
        """Create monitoring alert"""
        alert = MonitoringAlert(
            alert_id=f"alert_{uuid.uuid4().hex[:8]}",
            model_id=model_id,
            alert_type=alert_type,
            severity=severity,
            threshold=self.config['monitoring']['drift_threshold'],
            current_value=current_value,
            triggered_at=datetime.now(),
            message=message,
            resolved=False
        )
        
        # Store alert in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.alert_id, alert.model_id, alert.alert_type, alert.severity,
            alert.threshold, alert.current_value, alert.triggered_at.isoformat(),
            alert.message, alert.resolved
        ))
        
        conn.commit()
        conn.close()
        
        return alert
    
    # Additional helper methods would continue here...
    # This is a substantial enterprise platform with 2000+ lines when complete
