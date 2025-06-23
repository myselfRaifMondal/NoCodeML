"""
Enterprise AI-Powered Data Catalog and Discovery System

This module provides:
- Intelligent data discovery and profiling
- Automated data lineage tracking
- Schema evolution monitoring
- PII detection and classification
- Data quality scoring
- Semantic data understanding
- Automated documentation generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import uuid
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import hashlib
import re
from collections import Counter
import pickle
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)

class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class PIIType(Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    IP_ADDRESS = "ip_address"

class DataQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class DatasetProfile:
    """Comprehensive dataset profile"""
    dataset_id: str
    name: str
    description: str
    source_system: str
    created_at: datetime
    last_updated: datetime
    schema_version: str
    row_count: int
    column_count: int
    file_size_mb: float
    data_quality_score: float
    classification: DataClassification
    tags: List[str]
    business_glossary_terms: List[str]
    pii_detected: Dict[str, List[PIIType]]
    column_profiles: Dict[str, Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    lineage: Dict[str, Any]
    usage_stats: Dict[str, Any]

@dataclass
class ColumnProfile:
    """Detailed column profile"""
    column_name: str
    data_type: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    min_value: Any
    max_value: Any
    mean_value: Optional[float]
    std_value: Optional[float]
    top_values: List[Tuple[Any, int]]
    data_pattern: str
    semantic_type: str
    pii_types: List[PIIType]
    quality_issues: List[str]
    business_rules: List[str]

class IntelligentDataCatalog:
    """AI-powered data catalog and discovery system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.catalog_path = project_root / "data_catalog"
        self.profiles_path = self.catalog_path / "profiles"
        self.lineage_path = self.catalog_path / "lineage"
        
        # Initialize directories
        for path in [self.catalog_path, self.profiles_path, self.lineage_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = self.catalog_path / "catalog.db"
        self._init_database()
        
        # Initialize AI models for semantic understanding
        self._init_ai_models()
        
        # PII detection patterns
        self.pii_patterns = self._get_pii_patterns()
        
        # Business glossary
        self.business_glossary = self._load_business_glossary()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load data catalog configuration"""
        default_config = {
            "profiling": {
                "sample_size": 10000,
                "enable_semantic_analysis": True,
                "enable_pii_detection": True,
                "quality_thresholds": {
                    "excellent": 0.95,
                    "good": 0.85,
                    "fair": 0.70
                }
            },
            "lineage": {
                "auto_discovery": True,
                "track_transformations": True,
                "max_depth": 5
            },
            "privacy": {
                "auto_classify_pii": True,
                "mask_sensitive_data": True,
                "compliance_checks": ["GDPR", "CCPA", "HIPAA"]
            },
            "ai_features": {
                "auto_tagging": True,
                "semantic_search": True,
                "anomaly_detection": True,
                "recommendation_engine": True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def _init_database(self):
        """Initialize SQLite database for catalog metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Datasets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                source_system TEXT,
                created_at TIMESTAMP,
                last_updated TIMESTAMP,
                schema_version TEXT,
                row_count INTEGER,
                column_count INTEGER,
                file_size_mb REAL,
                data_quality_score REAL,
                classification TEXT,
                tags TEXT,
                business_glossary_terms TEXT,
                pii_detected TEXT,
                profile_data TEXT
            )
        """)
        
        # Columns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS columns (
                column_id TEXT PRIMARY KEY,
                dataset_id TEXT,
                column_name TEXT,
                data_type TEXT,
                semantic_type TEXT,
                null_percentage REAL,
                unique_percentage REAL,
                pii_types TEXT,
                quality_score REAL,
                profile_data TEXT,
                FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
            )
        """)
        
        # Data lineage table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_lineage (
                lineage_id TEXT PRIMARY KEY,
                source_dataset_id TEXT,
                target_dataset_id TEXT,
                transformation_type TEXT,
                transformation_logic TEXT,
                created_at TIMESTAMP,
                metadata TEXT
            )
        """)
        
        # Usage analytics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_analytics (
                usage_id TEXT PRIMARY KEY,
                dataset_id TEXT,
                user_id TEXT,
                access_type TEXT,
                access_timestamp TIMESTAMP,
                query_details TEXT,
                FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
            )
        """)
        
        # Data quality issues table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_issues (
                issue_id TEXT PRIMARY KEY,
                dataset_id TEXT,
                column_name TEXT,
                issue_type TEXT,
                severity TEXT,
                description TEXT,
                detected_at TIMESTAMP,
                resolved BOOLEAN,
                FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
            )
        """)
        
        # Schema evolution table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_evolution (
                evolution_id TEXT PRIMARY KEY,
                dataset_id TEXT,
                old_schema TEXT,
                new_schema TEXT,
                change_type TEXT,
                changed_at TIMESTAMP,
                impact_analysis TEXT,
                FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _init_ai_models(self):
        """Initialize AI models for semantic understanding"""
        try:
            # Initialize NLP models for text analysis
            import nltk
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            
            self.stop_words = set(stopwords.words('english'))
            
        except ImportError:
            logger.warning("NLTK not available for semantic analysis")
            self.stop_words = set()
    
    def _get_pii_patterns(self) -> Dict[PIIType, List[str]]:
        """Get regex patterns for PII detection"""
        return {
            PIIType.EMAIL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            PIIType.PHONE: [
                r'\b\d{3}-\d{3}-\d{4}\b',
                r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',
                r'\b\d{10}\b'
            ],
            PIIType.SSN: [
                r'\b\d{3}-\d{2}-\d{4}\b',
                r'\b\d{9}\b'
            ],
            PIIType.CREDIT_CARD: [
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
            ],
            PIIType.IP_ADDRESS: [
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            ]
        }
    
    def _load_business_glossary(self) -> Dict[str, Dict[str, str]]:
        """Load business glossary terms"""
        glossary_path = self.catalog_path / "business_glossary.json"
        
        if glossary_path.exists():
            with open(glossary_path, 'r') as f:
                return json.load(f)
        
        # Default business glossary
        default_glossary = {
            "customer": {
                "definition": "Individual or organization that purchases products or services",
                "synonyms": ["client", "buyer", "purchaser"],
                "related_terms": ["prospect", "lead", "account"]
            },
            "revenue": {
                "definition": "Income generated from business operations",
                "synonyms": ["income", "sales", "earnings"],
                "related_terms": ["profit", "margin", "cost"]
            },
            "transaction": {
                "definition": "Business event that has monetary impact",
                "synonyms": ["purchase", "sale", "payment"],
                "related_terms": ["order", "invoice", "receipt"]
            }
        }
        
        # Save default glossary
        with open(glossary_path, 'w') as f:
            json.dump(default_glossary, f, indent=2)
        
        return default_glossary
    
    async def catalog_dataset(self, data: pd.DataFrame, dataset_name: str,
                            source_system: str = "unknown",
                            description: str = "") -> str:
        """Catalog a dataset with comprehensive profiling"""
        try:
            dataset_id = f"dataset_{uuid.uuid4().hex[:8]}"
            
            logger.info(f"Starting comprehensive profiling for dataset: {dataset_name}")
            
            # Basic dataset information
            row_count = len(data)
            column_count = len(data.columns)
            file_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
            
            # Perform comprehensive profiling
            profile_results = await self._comprehensive_profiling(data, dataset_id)
            
            # PII detection
            pii_detected = await self._detect_pii(data)
            
            # Data quality assessment
            quality_score = await self._assess_data_quality(data, profile_results)
            
            # Automatic classification
            classification = await self._classify_dataset(data, pii_detected)
            
            # Semantic analysis and tagging
            tags = await self._generate_semantic_tags(data, dataset_name, description)
            
            # Business glossary mapping
            business_terms = await self._map_business_glossary(data, dataset_name, description)
            
            # Column profiles
            column_profiles = {}
            for column in data.columns:
                column_profiles[column] = await self._profile_column(data[column], column)
            
            # Create dataset profile
            dataset_profile = DatasetProfile(
                dataset_id=dataset_id,
                name=dataset_name,
                description=description,
                source_system=source_system,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                schema_version="1.0",
                row_count=row_count,
                column_count=column_count,
                file_size_mb=file_size_mb,
                data_quality_score=quality_score,
                classification=classification,
                tags=tags,
                business_glossary_terms=business_terms,
                pii_detected=pii_detected,
                column_profiles=column_profiles,
                relationships=[],
                lineage={},
                usage_stats={}
            )
            
            # Store in database
            await self._store_dataset_profile(dataset_profile)
            
            # Save detailed profile
            profile_file = self.profiles_path / f"{dataset_id}_profile.json"
            with open(profile_file, 'w') as f:
                # Convert non-serializable objects
                profile_dict = asdict(dataset_profile)
                profile_dict['created_at'] = dataset_profile.created_at.isoformat()
                profile_dict['last_updated'] = dataset_profile.last_updated.isoformat()
                profile_dict['classification'] = dataset_profile.classification.value
                json.dump(profile_dict, f, indent=2, default=str)
            
            logger.info(f"Dataset {dataset_name} cataloged successfully with ID: {dataset_id}")
            return dataset_id
            
        except Exception as e:
            logger.error(f"Failed to catalog dataset: {str(e)}")
            raise
    
    async def _comprehensive_profiling(self, data: pd.DataFrame, dataset_id: str) -> Dict[str, Any]:
        """Perform comprehensive data profiling"""
        profiling_results = {
            'basic_stats': {},
            'data_types': {},
            'missing_data': {},
            'uniqueness': {},
            'patterns': {},
            'correlations': {},
            'anomalies': []
        }
        
        # Basic statistics
        profiling_results['basic_stats'] = {
            'row_count': len(data),
            'column_count': len(data.columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024),
            'duplicates': data.duplicated().sum()
        }
        
        # Data types analysis
        profiling_results['data_types'] = {
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(data.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(data.select_dtypes(include=['datetime64']).columns),
            'boolean_columns': len(data.select_dtypes(include=['bool']).columns)
        }
        
        # Missing data analysis
        missing_data = data.isnull().sum()
        profiling_results['missing_data'] = {
            'total_missing': missing_data.sum(),
            'missing_percentage': (missing_data.sum() / (len(data) * len(data.columns))) * 100,
            'columns_with_missing': missing_data[missing_data > 0].to_dict()
        }
        
        # Uniqueness analysis
        profiling_results['uniqueness'] = {}
        for column in data.columns:
            unique_count = data[column].nunique()
            profiling_results['uniqueness'][column] = {
                'unique_count': unique_count,
                'unique_percentage': (unique_count / len(data)) * 100
            }
        
        # Pattern analysis for string columns
        for column in data.select_dtypes(include=['object']).columns:
            patterns = await self._analyze_patterns(data[column])
            profiling_results['patterns'][column] = patterns
        
        # Correlation analysis for numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            correlation_matrix = numeric_data.corr()
            high_correlations = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:
                        high_correlations.append({
                            'column1': correlation_matrix.columns[i],
                            'column2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            profiling_results['correlations'] = high_correlations
        
        # Anomaly detection
        anomalies = await self._detect_anomalies(data)
        profiling_results['anomalies'] = anomalies
        
        return profiling_results
    
    async def _detect_pii(self, data: pd.DataFrame) -> Dict[str, List[PIIType]]:
        """Detect PII in dataset columns"""
        pii_detected = {}
        
        for column in data.columns:
            column_pii = []
            sample_data = data[column].dropna().astype(str).head(1000)
            
            for pii_type, patterns in self.pii_patterns.items():
                for pattern in patterns:
                    matches = sample_data.str.contains(pattern, regex=True, na=False)
                    if matches.any():
                        # Calculate confidence based on match percentage
                        match_percentage = matches.sum() / len(sample_data)
                        if match_percentage > 0.1:  # At least 10% matches
                            column_pii.append(pii_type)
                        break
            
            # Additional semantic PII detection
            column_lower = column.lower()
            if any(keyword in column_lower for keyword in ['name', 'first', 'last']):
                if PIIType.NAME not in column_pii:
                    column_pii.append(PIIType.NAME)
            
            if any(keyword in column_lower for keyword in ['email', 'mail']):
                if PIIType.EMAIL not in column_pii:
                    column_pii.append(PIIType.EMAIL)
            
            if any(keyword in column_lower for keyword in ['phone', 'mobile', 'tel']):
                if PIIType.PHONE not in column_pii:
                    column_pii.append(PIIType.PHONE)
            
            if any(keyword in column_lower for keyword in ['address', 'street', 'city']):
                if PIIType.ADDRESS not in column_pii:
                    column_pii.append(PIIType.ADDRESS)
            
            if column_pii:
                pii_detected[column] = column_pii
        
        return pii_detected
    
    async def _assess_data_quality(self, data: pd.DataFrame, 
                                 profile_results: Dict[str, Any]) -> float:
        """Assess overall data quality score"""
        quality_factors = []
        
        # Completeness (missing data)
        completeness = 1 - (profile_results['missing_data']['missing_percentage'] / 100)
        quality_factors.append(('completeness', completeness, 0.3))
        
        # Uniqueness (duplicates)
        uniqueness = 1 - (profile_results['basic_stats']['duplicates'] / len(data))
        quality_factors.append(('uniqueness', uniqueness, 0.2))
        
        # Consistency (data type consistency)
        consistency_score = 1.0  # Default to perfect consistency
        for column in data.columns:
            if data[column].dtype == 'object':
                # Check for mixed types in object columns
                sample = data[column].dropna().head(100)
                type_consistency = len(set(type(x).__name__ for x in sample)) == 1
                if not type_consistency:
                    consistency_score -= 0.1
        
        consistency_score = max(0, consistency_score)
        quality_factors.append(('consistency', consistency_score, 0.2))
        
        # Validity (pattern conformance)
        validity_scores = []
        for column in data.select_dtypes(include=['object']).columns:
            patterns = profile_results['patterns'].get(column, {})
            if patterns.get('dominant_pattern_percentage', 0) > 80:
                validity_scores.append(0.9)
            elif patterns.get('dominant_pattern_percentage', 0) > 60:
                validity_scores.append(0.7)
            else:
                validity_scores.append(0.5)
        
        validity = np.mean(validity_scores) if validity_scores else 1.0
        quality_factors.append(('validity', validity, 0.15))
        
        # Accuracy (outlier detection)
        accuracy_score = 1 - (len(profile_results['anomalies']) / len(data))
        accuracy_score = max(0, accuracy_score)
        quality_factors.append(('accuracy', accuracy_score, 0.15))
        
        # Calculate weighted score
        total_score = sum(score * weight for _, score, weight in quality_factors)
        
        return round(total_score, 3)
    
    async def _classify_dataset(self, data: pd.DataFrame, 
                              pii_detected: Dict[str, List[PIIType]]) -> DataClassification:
        """Automatically classify dataset based on PII and sensitivity"""
        
        # Check for restricted PII types
        restricted_pii = [PIIType.SSN, PIIType.CREDIT_CARD]
        if any(pii_type in pii_list for pii_list in pii_detected.values() 
               for pii_type in restricted_pii):
            return DataClassification.RESTRICTED
        
        # Check for confidential PII types
        confidential_pii = [PIIType.EMAIL, PIIType.PHONE, PIIType.ADDRESS, PIIType.DATE_OF_BIRTH]
        if any(pii_type in pii_list for pii_list in pii_detected.values() 
               for pii_type in confidential_pii):
            return DataClassification.CONFIDENTIAL
        
        # Check for any PII
        if pii_detected:
            return DataClassification.INTERNAL
        
        # Default to public if no PII detected
        return DataClassification.PUBLIC
    
    async def _generate_semantic_tags(self, data: pd.DataFrame, 
                                     dataset_name: str, description: str) -> List[str]:
        """Generate semantic tags using AI"""
        tags = set()
        
        # Extract tags from dataset name
        name_words = re.findall(r'\w+', dataset_name.lower())
        tags.update(word for word in name_words if len(word) > 2)
        
        # Extract tags from description
        if description:
            desc_words = re.findall(r'\w+', description.lower())
            tags.update(word for word in desc_words if len(word) > 3 and word not in self.stop_words)
        
        # Analyze column names for domain-specific tags
        column_domains = {
            'financial': ['price', 'cost', 'revenue', 'profit', 'amount', 'balance', 'payment'],
            'customer': ['customer', 'client', 'user', 'buyer', 'name', 'email', 'phone'],
            'temporal': ['date', 'time', 'timestamp', 'created', 'updated', 'modified'],
            'geographic': ['address', 'city', 'state', 'country', 'zip', 'postal', 'location'],
            'product': ['product', 'item', 'category', 'brand', 'model', 'sku'],
            'transaction': ['order', 'transaction', 'purchase', 'sale', 'invoice']
        }
        
        for domain, keywords in column_domains.items():
            column_text = ' '.join(data.columns).lower()
            if any(keyword in column_text for keyword in keywords):
                tags.add(domain)
        
        # Add data type tags
        if len(data.select_dtypes(include=[np.number]).columns) > len(data.columns) * 0.5:
            tags.add('numerical')
        
        if len(data.select_dtypes(include=['object']).columns) > len(data.columns) * 0.5:
            tags.add('categorical')
        
        if len(data.select_dtypes(include=['datetime64']).columns) > 0:
            tags.add('time_series')
        
        return list(tags)
    
    async def _map_business_glossary(self, data: pd.DataFrame, 
                                   dataset_name: str, description: str) -> List[str]:
        """Map dataset to business glossary terms"""
        mapped_terms = []
        
        # Text to analyze
        text_to_analyze = f"{dataset_name} {description} {' '.join(data.columns)}".lower()
        
        # Find matching glossary terms
        for term, term_info in self.business_glossary.items():
            # Check direct term match
            if term.lower() in text_to_analyze:
                mapped_terms.append(term)
                continue
            
            # Check synonyms
            synonyms = term_info.get('synonyms', [])
            if any(synonym.lower() in text_to_analyze for synonym in synonyms):
                mapped_terms.append(term)
                continue
            
            # Check related terms
            related_terms = term_info.get('related_terms', [])
            if any(related.lower() in text_to_analyze for related in related_terms):
                mapped_terms.append(term)
        
        return mapped_terms
    
    async def _profile_column(self, column_data: pd.Series, column_name: str) -> Dict[str, Any]:
        """Create detailed profile for a single column"""
        profile = {
            'data_type': str(column_data.dtype),
            'null_count': column_data.isnull().sum(),
            'null_percentage': (column_data.isnull().sum() / len(column_data)) * 100,
            'unique_count': column_data.nunique(),
            'unique_percentage': (column_data.nunique() / len(column_data)) * 100,
        }
        
        # Numeric column analysis
        if pd.api.types.is_numeric_dtype(column_data):
            profile.update({
                'min_value': column_data.min(),
                'max_value': column_data.max(),
                'mean_value': column_data.mean(),
                'median_value': column_data.median(),
                'std_value': column_data.std(),
                'percentiles': {
                    '25': column_data.quantile(0.25),
                    '50': column_data.quantile(0.50),
                    '75': column_data.quantile(0.75),
                    '95': column_data.quantile(0.95)
                }
            })
        
        # Categorical column analysis
        elif column_data.dtype == 'object':
            value_counts = column_data.value_counts().head(10)
            profile.update({
                'top_values': list(zip(value_counts.index, value_counts.values)),
                'avg_length': column_data.astype(str).str.len().mean(),
                'max_length': column_data.astype(str).str.len().max(),
                'min_length': column_data.astype(str).str.len().min()
            })
        
        # DateTime column analysis
        elif pd.api.types.is_datetime64_any_dtype(column_data):
            profile.update({
                'min_date': column_data.min(),
                'max_date': column_data.max(),
                'date_range_days': (column_data.max() - column_data.min()).days
            })
        
        # Semantic type inference
        profile['semantic_type'] = await self._infer_semantic_type(column_data, column_name)
        
        return profile
    
    async def _infer_semantic_type(self, column_data: pd.Series, column_name: str) -> str:
        """Infer semantic type of column"""
        column_lower = column_name.lower()
        
        # Check for common semantic types based on name
        if any(keyword in column_lower for keyword in ['id', 'key', 'pk']):
            return 'identifier'
        elif any(keyword in column_lower for keyword in ['name', 'title']):
            return 'name'
        elif any(keyword in column_lower for keyword in ['email', 'mail']):
            return 'email'
        elif any(keyword in column_lower for keyword in ['phone', 'mobile']):
            return 'phone'
        elif any(keyword in column_lower for keyword in ['address', 'street']):
            return 'address'
        elif any(keyword in column_lower for keyword in ['date', 'time', 'created', 'updated']):
            return 'datetime'
        elif any(keyword in column_lower for keyword in ['price', 'cost', 'amount', 'value']):
            return 'monetary'
        elif any(keyword in column_lower for keyword in ['percent', 'rate', 'ratio']):
            return 'percentage'
        elif any(keyword in column_lower for keyword in ['count', 'quantity', 'number']):
            return 'count'
        elif any(keyword in column_lower for keyword in ['category', 'type', 'class']):
            return 'category'
        elif any(keyword in column_lower for keyword in ['url', 'link', 'website']):
            return 'url'
        
        # Analyze data patterns for additional inference
        if column_data.dtype == 'object':
            sample_data = column_data.dropna().astype(str).head(100)
            
            # Check for URL pattern
            if sample_data.str.contains(r'https?://', regex=True).any():
                return 'url'
            
            # Check for email pattern
            if sample_data.str.contains(r'@.*\.', regex=True).any():
                return 'email'
            
            # Check for numeric ID pattern
            if sample_data.str.isdigit().all():
                return 'numeric_id'
        
        # Default semantic types based on data type
        if pd.api.types.is_numeric_dtype(column_data):
            if column_data.nunique() == len(column_data):
                return 'numeric_id'
            else:
                return 'numeric_measure'
        elif pd.api.types.is_datetime64_any_dtype(column_data):
            return 'datetime'
        elif column_data.dtype == 'bool':
            return 'boolean'
        else:
            return 'text'
    
    async def _analyze_patterns(self, column_data: pd.Series) -> Dict[str, Any]:
        """Analyze patterns in string column"""
        patterns = {}
        
        # Convert to string and get non-null values
        string_data = column_data.dropna().astype(str)
        
        if len(string_data) == 0:
            return patterns
        
        # Length analysis
        lengths = string_data.str.len()
        patterns['length_stats'] = {
            'min': lengths.min(),
            'max': lengths.max(),
            'mean': lengths.mean(),
            'std': lengths.std()
        }
        
        # Character composition analysis
        patterns['character_composition'] = {
            'alpha_only': string_data.str.isalpha().sum(),
            'numeric_only': string_data.str.isnumeric().sum(),
            'alphanumeric': string_data.str.isalnum().sum(),
            'has_spaces': string_data.str.contains(' ').sum(),
            'has_special_chars': string_data.str.contains(r'[^a-zA-Z0-9\s]').sum()
        }
        
        # Case analysis
        patterns['case_analysis'] = {
            'upper_case': string_data.str.isupper().sum(),
            'lower_case': string_data.str.islower().sum(),
            'title_case': string_data.str.istitle().sum()
        }
        
        # Common format patterns
        format_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone_us': r'^\d{3}-\d{3}-\d{4}$',
            'ssn': r'^\d{3}-\d{2}-\d{4}$',
            'credit_card': r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$',
            'zip_code': r'^\d{5}(-\d{4})?$',
            'date_iso': r'^\d{4}-\d{2}-\d{2}$',
            'url': r'^https?://.*',
            'ip_address': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        }
        
        patterns['format_matches'] = {}
        for pattern_name, pattern in format_patterns.items():
            matches = string_data.str.match(pattern).sum()
            if matches > 0:
                patterns['format_matches'][pattern_name] = {
                    'count': matches,
                    'percentage': (matches / len(string_data)) * 100
                }
        
        # Find dominant pattern
        if patterns['format_matches']:
            dominant_pattern = max(patterns['format_matches'].items(), 
                                 key=lambda x: x[1]['percentage'])
            patterns['dominant_pattern'] = dominant_pattern[0]
            patterns['dominant_pattern_percentage'] = dominant_pattern[1]['percentage']
        
        return patterns
    
    async def _detect_anomalies(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalies in the dataset"""
        anomalies = []
        
        # Numeric anomalies using IQR method
        for column in data.select_dtypes(include=[np.number]).columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # Avoid division by zero
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
                
                if len(outliers) > 0:
                    anomalies.append({
                        'type': 'statistical_outlier',
                        'column': column,
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(data)) * 100,
                        'bounds': {'lower': lower_bound, 'upper': upper_bound}
                    })
        
        # String length anomalies
        for column in data.select_dtypes(include=['object']).columns:
            if data[column].dtype == 'object':
                lengths = data[column].astype(str).str.len()
                mean_length = lengths.mean()
                std_length = lengths.std()
                
                if std_length > 0:
                    z_scores = np.abs((lengths - mean_length) / std_length)
                    length_outliers = data[z_scores > 3]
                    
                    if len(length_outliers) > 0:
                        anomalies.append({
                            'type': 'length_anomaly',
                            'column': column,
                            'count': len(length_outliers),
                            'percentage': (len(length_outliers) / len(data)) * 100
                        })
        
        # Cardinality anomalies
        for column in data.columns:
            unique_ratio = data[column].nunique() / len(data)
            
            if unique_ratio > 0.95 and len(data) > 100:  # High cardinality
                anomalies.append({
                    'type': 'high_cardinality',
                    'column': column,
                    'unique_ratio': unique_ratio,
                    'message': 'Column has unusually high cardinality'
                })
            elif unique_ratio < 0.01 and data[column].nunique() > 1:  # Low cardinality
                anomalies.append({
                    'type': 'low_cardinality',
                    'column': column,
                    'unique_ratio': unique_ratio,
                    'message': 'Column has unusually low cardinality'
                })
        
        return anomalies
    
    async def discover_relationships(self, dataset_ids: List[str]) -> List[Dict[str, Any]]:
        """Discover relationships between datasets"""
        relationships = []
        
        # Load dataset profiles
        profiles = {}
        for dataset_id in dataset_ids:
            profile = await self.get_dataset_profile(dataset_id)
            if profile:
                profiles[dataset_id] = profile
        
        # Compare datasets pairwise
        for i, dataset_id_a in enumerate(dataset_ids):
            for dataset_id_b in dataset_ids[i+1:]:
                if dataset_id_a in profiles and dataset_id_b in profiles:
                    relationship = await self._analyze_dataset_relationship(
                        profiles[dataset_id_a], profiles[dataset_id_b]
                    )
                    if relationship:
                        relationships.append(relationship)
        
        return relationships
    
    async def _analyze_dataset_relationship(self, profile_a: Dict[str, Any], 
                                          profile_b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze relationship between two datasets"""
        
        # Check for common columns
        columns_a = set(profile_a['column_profiles'].keys())
        columns_b = set(profile_b['column_profiles'].keys())
        common_columns = columns_a & columns_b
        
        if not common_columns:
            return None
        
        # Analyze common columns
        relationship_strength = 0
        column_relationships = []
        
        for column in common_columns:
            col_profile_a = profile_a['column_profiles'][column]
            col_profile_b = profile_b['column_profiles'][column]
            
            # Check data type similarity
            if col_profile_a['data_type'] == col_profile_b['data_type']:
                relationship_strength += 1
            
            # Check semantic type similarity
            if col_profile_a.get('semantic_type') == col_profile_b.get('semantic_type'):
                relationship_strength += 1
            
            column_relationships.append({
                'column': column,
                'type_match': col_profile_a['data_type'] == col_profile_b['data_type'],
                'semantic_match': col_profile_a.get('semantic_type') == col_profile_b.get('semantic_type')
            })
        
        # Determine relationship type
        overlap_ratio = len(common_columns) / min(len(columns_a), len(columns_b))
        
        if overlap_ratio > 0.8:
            relationship_type = 'duplicate_or_version'
        elif overlap_ratio > 0.5:
            relationship_type = 'strong_overlap'
        elif overlap_ratio > 0.2:
            relationship_type = 'moderate_overlap'
        else:
            relationship_type = 'weak_overlap'
        
        return {
            'dataset_a_id': profile_a['dataset_id'],
            'dataset_b_id': profile_b['dataset_id'],
            'relationship_type': relationship_type,
            'common_columns': list(common_columns),
            'overlap_ratio': overlap_ratio,
            'relationship_strength': relationship_strength,
            'column_relationships': column_relationships
        }
    
    async def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search across cataloged datasets"""
        search_results = []
        
        # Load all dataset profiles
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT dataset_id, name, description, tags, business_glossary_terms 
            FROM datasets
        """)
        
        datasets = cursor.fetchall()
        conn.close()
        
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        for dataset in datasets:
            dataset_id, name, description, tags, business_terms = dataset
            
            # Calculate relevance score
            relevance_score = 0
            
            # Name matching
            if query_lower in name.lower():
                relevance_score += 10
            
            name_words = set(re.findall(r'\w+', name.lower()))
            relevance_score += len(query_words & name_words) * 5
            
            # Description matching
            if description and query_lower in description.lower():
                relevance_score += 8
            
            if description:
                desc_words = set(re.findall(r'\w+', description.lower()))
                relevance_score += len(query_words & desc_words) * 3
            
            # Tags matching
            if tags:
                tag_list = json.loads(tags) if isinstance(tags, str) else tags
                tag_words = set(tag.lower() for tag in tag_list)
                relevance_score += len(query_words & tag_words) * 4
            
            # Business terms matching
            if business_terms:
                term_list = json.loads(business_terms) if isinstance(business_terms, str) else business_terms
                term_words = set(term.lower() for term in term_list)
                relevance_score += len(query_words & term_words) * 6
            
            if relevance_score > 0:
                search_results.append({
                    'dataset_id': dataset_id,
                    'name': name,
                    'description': description,
                    'relevance_score': relevance_score,
                    'tags': json.loads(tags) if tags else [],
                    'business_terms': json.loads(business_terms) if business_terms else []
                })
        
        # Sort by relevance score and limit results
        search_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return search_results[:limit]
    
    async def generate_data_documentation(self, dataset_id: str) -> str:
        """Generate comprehensive data documentation"""
        profile = await self.get_dataset_profile(dataset_id)
        if not profile:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        doc = f"""
# Data Documentation: {profile['name']}

## Overview
**Dataset ID:** {profile['dataset_id']}
**Source System:** {profile['source_system']}
**Created:** {profile['created_at']}
**Last Updated:** {profile['last_updated']}
**Classification:** {profile['classification']}

## Description
{profile['description'] or 'No description provided'}

## Dataset Statistics
- **Rows:** {profile['row_count']:,}
- **Columns:** {profile['column_count']}
- **File Size:** {profile['file_size_mb']:.2f} MB
- **Data Quality Score:** {profile['data_quality_score']:.2%}

## Tags
{', '.join(profile['tags']) if profile['tags'] else 'No tags assigned'}

## Business Glossary Terms
{', '.join(profile['business_glossary_terms']) if profile['business_glossary_terms'] else 'No business terms mapped'}

## Privacy and Compliance
"""
        
        if profile['pii_detected']:
            doc += "\n### PII Detected\n"
            for column, pii_types in profile['pii_detected'].items():
                pii_names = [pii_type.value for pii_type in pii_types]
                doc += f"- **{column}:** {', '.join(pii_names)}\n"
        else:
            doc += "\n### PII Status\nNo PII detected in this dataset.\n"
        
        doc += "\n## Column Profiles\n"
        
        for column_name, column_profile in profile['column_profiles'].items():
            doc += f"\n### {column_name}\n"
            doc += f"- **Data Type:** {column_profile['data_type']}\n"
            doc += f"- **Semantic Type:** {column_profile.get('semantic_type', 'Unknown')}\n"
            doc += f"- **Null Percentage:** {column_profile['null_percentage']:.1f}%\n"
            doc += f"- **Unique Values:** {column_profile['unique_count']} ({column_profile['unique_percentage']:.1f}%)\n"
            
            if 'min_value' in column_profile:
                doc += f"- **Range:** {column_profile['min_value']} to {column_profile['max_value']}\n"
                doc += f"- **Mean:** {column_profile.get('mean_value', 'N/A')}\n"
            
            if 'top_values' in column_profile:
                doc += f"- **Top Values:** {column_profile['top_values'][:5]}\n"
        
        return doc
    
    # Helper methods
    async def get_dataset_profile(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve dataset profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            profile = dict(zip(columns, row))
            
            # Parse JSON fields
            for field in ['tags', 'business_glossary_terms', 'pii_detected']:
                if profile[field]:
                    profile[field] = json.loads(profile[field])
            
            # Load detailed profile
            profile_file = self.profiles_path / f"{dataset_id}_profile.json"
            if profile_file.exists():
                with open(profile_file, 'r') as f:
                    detailed_profile = json.load(f)
                profile.update(detailed_profile)
            
            return profile
        return None
    
    async def _store_dataset_profile(self, profile: DatasetProfile):
        """Store dataset profile in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO datasets VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            profile.dataset_id, profile.name, profile.description, profile.source_system,
            profile.created_at.isoformat(), profile.last_updated.isoformat(),
            profile.schema_version, profile.row_count, profile.column_count,
            profile.file_size_mb, profile.data_quality_score, profile.classification.value,
            json.dumps(profile.tags), json.dumps(profile.business_glossary_terms),
            json.dumps(profile.pii_detected, default=lambda x: x.value if hasattr(x, 'value') else str(x)),
            json.dumps(profile.column_profiles, default=str)
        ))
        
        conn.commit()
        conn.close()
