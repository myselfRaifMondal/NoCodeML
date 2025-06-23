"""
Enterprise Intelligent Visualization Tools Module

This module provides advanced visualization capabilities for data analysis,
model insights, performance monitoring, and interactive dashboards.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path
import base64
import io
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import networkx as nx
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class VisualizationType(Enum):
    DATA_DISTRIBUTION = "data_distribution"
    CORRELATION_MATRIX = "correlation_matrix"
    TIME_SERIES = "time_series"
    MODEL_PERFORMANCE = "model_performance"
    FEATURE_IMPORTANCE = "feature_importance"
    BIAS_ANALYSIS = "bias_analysis"
    DATA_LINEAGE = "data_lineage"
    CLUSTER_ANALYSIS = "cluster_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    COMPARATIVE_ANALYSIS = "comparative_analysis"

class ChartType(Enum):
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    TREEMAP = "treemap"
    SANKEY = "sankey"
    NETWORK = "network"
    THREE_D = "3d"

@dataclass
class VisualizationConfig:
    id: str
    title: str
    type: VisualizationType
    chart_type: ChartType
    data_source: str
    parameters: Dict[str, Any]
    styling: Dict[str, Any]
    interactive: bool
    created_at: datetime
    updated_at: datetime

@dataclass
class VisualizationInsight:
    id: str
    visualization_id: str
    insight_type: str
    description: str
    confidence: float
    data_points: List[Any]
    recommendations: List[str]
    timestamp: datetime

class IntelligentVisualizationEngine:
    """
    Advanced visualization engine with AI-powered insights and recommendations.
    """
    
    def __init__(self, db_path: str = "visualizations.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # Color palettes for different themes
        self.color_palettes = {
            "enterprise": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
            "performance": ["#28a745", "#ffc107", "#dc3545", "#17a2b8", "#6f42c1"],
            "bias": ["#fd7e14", "#20c997", "#6610f2", "#e83e8c", "#6c757d"],
            "quality": ["#007bff", "#28a745", "#ffc107", "#dc3545"],
            "modern": ["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe", "#00f2fe"]
        }
        
        self._init_database()
    
    def _init_database(self):
        """Initialize the visualization database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Visualization configurations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visualizations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                chart_type TEXT NOT NULL,
                data_source TEXT NOT NULL,
                parameters TEXT NOT NULL,
                styling TEXT NOT NULL,
                interactive BOOLEAN DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Visualization insights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visualization_insights (
                id TEXT PRIMARY KEY,
                visualization_id TEXT NOT NULL,
                insight_type TEXT NOT NULL,
                description TEXT NOT NULL,
                confidence REAL NOT NULL,
                data_points TEXT NOT NULL,
                recommendations TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (visualization_id) REFERENCES visualizations (id)
            )
        ''')
        
        # Visualization cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visualization_cache (
                id TEXT PRIMARY KEY,
                visualization_id TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                FOREIGN KEY (visualization_id) REFERENCES visualizations (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_data_distribution_analysis(self, data: pd.DataFrame, title: str = "Data Distribution Analysis") -> Dict[str, Any]:
        """Create comprehensive data distribution analysis with multiple visualizations."""
        
        # Create subplots for comprehensive analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribution Overview', 'Statistical Summary', 'Correlation Heatmap', 'Missing Data Analysis'),
            specs=[[{"secondary_y": True}, {"type": "table"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        # 1. Distribution Overview
        for i, col in enumerate(numeric_cols[:5]):  # Limit to first 5 numeric columns
            fig.add_trace(
                go.Histogram(
                    x=data[col],
                    name=col,
                    opacity=0.7,
                    nbinsx=30
                ),
                row=1, col=1
            )
        
        # 2. Statistical Summary Table
        summary_stats = data.describe().round(3)
        fig.add_trace(
            go.Table(
                header=dict(values=['Statistic'] + list(summary_stats.columns),
                           fill_color='lightblue',
                           align='center'),
                cells=dict(values=[summary_stats.index] + [summary_stats[col] for col in summary_stats.columns],
                          fill_color='white',
                          align='center')
            ),
            row=1, col=2
        )
        
        # 3. Correlation Heatmap
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.round(2).values,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    showscale=False
                ),
                row=2, col=1
            )
        
        # 4. Missing Data Analysis
        missing_data = data.isnull().sum()
        missing_percentage = (missing_data / len(data)) * 100
        
        fig.add_trace(
            go.Bar(
                x=missing_percentage.index,
                y=missing_percentage.values,
                name='Missing %',
                marker_color='red',
                opacity=0.7
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            showlegend=True,
            height=800,
            template="plotly_white"
        )
        
        # Generate insights
        insights = self._generate_distribution_insights(data)
        
        return {
            "figure": fig,
            "insights": insights,
            "summary": {
                "total_records": len(data),
                "numeric_features": len(numeric_cols),
                "categorical_features": len(categorical_cols),
                "missing_data_percentage": (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            }
        }
    
    def create_model_performance_dashboard(self, model_metrics: Dict[str, Any], 
                                         predictions: np.ndarray, 
                                         actual: np.ndarray,
                                         title: str = "Model Performance Dashboard") -> Dict[str, Any]:
        """Create comprehensive model performance visualization."""
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Performance Metrics', 'Confusion Matrix', 'ROC Curve',
                          'Prediction Distribution', 'Residual Analysis', 'Feature Importance'),
            specs=[[{"type": "bar"}, {"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Performance Metrics Bar Chart
        metrics_names = list(model_metrics.keys())
        metrics_values = list(model_metrics.values())
        
        colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in metrics_values]
        
        fig.add_trace(
            go.Bar(
                x=metrics_names,
                y=metrics_values,
                marker_color=colors,
                text=[f"{v:.3f}" for v in metrics_values],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Confusion Matrix (for classification)
        if len(np.unique(actual)) < 20:  # Assume classification if few unique values
            cm = confusion_matrix(actual, predictions)
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    colorscale='Blues',
                    showscale=False
                ),
                row=1, col=2
            )
        
        # 3. ROC Curve (for binary classification)
        if len(np.unique(actual)) == 2:
            fpr, tpr, _ = roc_curve(actual, predictions)
            auc_score = auc(fpr, tpr)
            
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'ROC (AUC = {auc_score:.3f})',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=3
            )
            
            # Add diagonal line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=3
            )
        
        # 4. Prediction Distribution
        fig.add_trace(
            go.Histogram(
                x=predictions,
                name='Predictions',
                opacity=0.7,
                nbinsx=30,
                marker_color='blue'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=actual,
                name='Actual',
                opacity=0.7,
                nbinsx=30,
                marker_color='red'
            ),
            row=2, col=1
        )
        
        # 5. Residual Analysis (for regression)
        if len(np.unique(actual)) > 20:  # Assume regression
            residuals = actual - predictions
            fig.add_trace(
                go.Scatter(
                    x=predictions,
                    y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(color='purple', opacity=0.6)
                ),
                row=2, col=2
            )
            
            # Add zero line
            fig.add_trace(
                go.Scatter(
                    x=[predictions.min(), predictions.max()],
                    y=[0, 0],
                    mode='lines',
                    name='Zero Line',
                    line=dict(color='black', dash='dash')
                ),
                row=2, col=2
            )
        
        # 6. Feature Importance (if provided)
        if 'feature_importance' in model_metrics:
            importance_data = model_metrics['feature_importance']
            if isinstance(importance_data, dict):
                features = list(importance_data.keys())
                importance = list(importance_data.values())
                
                fig.add_trace(
                    go.Bar(
                        x=importance,
                        y=features,
                        orientation='h',
                        marker_color='green',
                        opacity=0.7
                    ),
                    row=2, col=3
                )
        
        fig.update_layout(
            title=title,
            showlegend=True,
            height=900,
            template="plotly_white"
        )
        
        # Generate performance insights
        insights = self._generate_performance_insights(model_metrics, predictions, actual)
        
        return {
            "figure": fig,
            "insights": insights,
            "metrics": model_metrics
        }
    
    def create_bias_analysis_visualization(self, data: pd.DataFrame, 
                                         protected_attributes: List[str],
                                         target_column: str,
                                         predictions: Optional[np.ndarray] = None,
                                         title: str = "Bias Analysis Dashboard") -> Dict[str, Any]:
        """Create comprehensive bias analysis visualization."""
        
        n_attributes = len(protected_attributes)
        rows = (n_attributes + 1) // 2 + 1
        
        fig = make_subplots(
            rows=rows, cols=2,
            subplot_titles=['Bias Overview'] + [f'Bias Analysis: {attr}' for attr in protected_attributes],
            specs=[[{"colspan": 2}] + [[{}, {}] for _ in range(rows-1)]]
        )
        
        bias_metrics = {}
        
        # 1. Overall Bias Overview
        overall_bias_data = []
        for attr in protected_attributes:
            if attr in data.columns:
                groups = data[attr].unique()
                for group in groups:
                    group_data = data[data[attr] == group]
                    positive_rate = (group_data[target_column] == 1).mean() if predictions is None else (predictions[data[attr] == group] == 1).mean()
                    overall_bias_data.append({
                        'Attribute': attr,
                        'Group': str(group),
                        'Positive_Rate': positive_rate,
                        'Count': len(group_data)
                    })
        
        bias_df = pd.DataFrame(overall_bias_data)
        
        fig.add_trace(
            go.Bar(
                x=[f"{row['Attribute']}: {row['Group']}" for _, row in bias_df.iterrows()],
                y=bias_df['Positive_Rate'],
                text=[f"{rate:.3f}" for rate in bias_df['Positive_Rate']],
                textposition='auto',
                marker_color=self.color_palettes['bias'][:len(bias_df)],
                name='Positive Rate by Group'
            ),
            row=1, col=1
        )
        
        # 2. Detailed Analysis for each Protected Attribute
        for i, attr in enumerate(protected_attributes):
            if attr not in data.columns:
                continue
                
            row_idx = (i // 2) + 2
            col_idx = (i % 2) + 1
            
            # Calculate bias metrics
            groups = data[attr].unique()
            group_stats = []
            
            for group in groups:
                group_mask = data[attr] == group
                group_data = data[group_mask]
                
                if predictions is not None:
                    group_predictions = predictions[group_mask]
                    positive_rate = (group_predictions == 1).mean()
                    accuracy = (group_predictions == group_data[target_column]).mean()
                else:
                    positive_rate = (group_data[target_column] == 1).mean()
                    accuracy = None
                
                group_stats.append({
                    'group': str(group),
                    'positive_rate': positive_rate,
                    'count': len(group_data),
                    'percentage': len(group_data) / len(data) * 100,
                    'accuracy': accuracy
                })
            
            # Demographic Parity calculation
            positive_rates = [stat['positive_rate'] for stat in group_stats]
            demographic_parity = max(positive_rates) - min(positive_rates)
            bias_metrics[f"{attr}_demographic_parity"] = demographic_parity
            
            # Create detailed visualization
            group_names = [stat['group'] for stat in group_stats]
            
            # Bar chart for positive rates
            fig.add_trace(
                go.Bar(
                    x=group_names,
                    y=positive_rates,
                    name=f'{attr} Positive Rate',
                    marker_color=self.color_palettes['bias'][i % len(self.color_palettes['bias'])],
                    opacity=0.7,
                    text=[f"{rate:.3f}" for rate in positive_rates],
                    textposition='auto'
                ),
                row=row_idx, col=col_idx
            )
            
            # Add demographic parity line
            avg_positive_rate = np.mean(positive_rates)
            fig.add_hline(
                y=avg_positive_rate,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Average: {avg_positive_rate:.3f}",
                row=row_idx, col=col_idx
            )
        
        fig.update_layout(
            title=title,
            showlegend=True,
            height=300 * rows,
            template="plotly_white"
        )
        
        # Generate bias insights
        insights = self._generate_bias_insights(bias_metrics, data, protected_attributes)
        
        return {
            "figure": fig,
            "bias_metrics": bias_metrics,
            "insights": insights
        }
    
    def create_data_lineage_visualization(self, lineage_data: Dict[str, Any], 
                                        title: str = "Data Lineage Visualization") -> Dict[str, Any]:
        """Create interactive data lineage network visualization."""
        
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes and edges from lineage data
        for source, targets in lineage_data.items():
            G.add_node(source)
            if isinstance(targets, list):
                for target in targets:
                    G.add_node(target)
                    G.add_edge(source, target)
            elif isinstance(targets, dict):
                for target, metadata in targets.items():
                    G.add_node(target)
                    G.add_edge(source, target, **metadata)
        
        # Calculate layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color='gray'),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Color nodes based on type (you can customize this based on your data)
            if 'dataset' in node.lower():
                node_colors.append('blue')
                node_sizes.append(20)
            elif 'model' in node.lower():
                node_colors.append('green')
                node_sizes.append(25)
            elif 'pipeline' in node.lower():
                node_colors.append('orange')
                node_sizes.append(15)
            else:
                node_colors.append('gray')
                node_sizes.append(10)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='middle center',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=[*edge_traces, node_trace])
        
        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Data Lineage Network",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white"
        )
        
        # Generate lineage insights
        insights = self._generate_lineage_insights(G, lineage_data)
        
        return {
            "figure": fig,
            "network_stats": {
                "nodes": len(G.nodes()),
                "edges": len(G.edges()),
                "density": nx.density(G),
                "components": nx.number_weakly_connected_components(G)
            },
            "insights": insights
        }
    
    def create_time_series_analysis(self, data: pd.DataFrame, 
                                  time_column: str, 
                                  value_columns: List[str],
                                  title: str = "Time Series Analysis") -> Dict[str, Any]:
        """Create comprehensive time series analysis."""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Time Series Overview', 'Trend Analysis', 
                          'Seasonality Analysis', 'Anomaly Detection',
                          'Distribution Changes', 'Forecast'),
            specs=[[{"colspan": 2}],
                   [{}, {}],
                   [{}, {}]]
        )
        
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
            data[time_column] = pd.to_datetime(data[time_column])
        
        data = data.sort_values(time_column)
        
        # 1. Time Series Overview
        for i, col in enumerate(value_columns[:5]):  # Limit to 5 columns
            fig.add_trace(
                go.Scatter(
                    x=data[time_column],
                    y=data[col],
                    mode='lines',
                    name=col,
                    line=dict(color=self.color_palettes['modern'][i % len(self.color_palettes['modern'])])
                ),
                row=1, col=1
            )
        
        # 2. Trend Analysis (moving averages)
        main_column = value_columns[0]
        
        # Calculate moving averages
        data['MA_7'] = data[main_column].rolling(window=7).mean()
        data['MA_30'] = data[main_column].rolling(window=30).mean()
        
        fig.add_trace(
            go.Scatter(
                x=data[time_column],
                y=data[main_column],
                mode='lines',
                name='Original',
                line=dict(color='blue', width=1),
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data[time_column],
                y=data['MA_7'],
                mode='lines',
                name='7-day MA',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data[time_column],
                y=data['MA_30'],
                mode='lines',
                name='30-day MA',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # 3. Seasonality Analysis (if enough data)
        if len(data) > 365:
            data['month'] = data[time_column].dt.month
            monthly_avg = data.groupby('month')[main_column].mean()
            
            fig.add_trace(
                go.Bar(
                    x=monthly_avg.index,
                    y=monthly_avg.values,
                    name='Monthly Average',
                    marker_color='purple',
                    opacity=0.7
                ),
                row=2, col=2
            )
        
        # 4. Anomaly Detection (simple statistical method)
        z_scores = np.abs(stats.zscore(data[main_column].dropna()))
        anomalies = data[z_scores > 3]
        
        fig.add_trace(
            go.Scatter(
                x=data[time_column],
                y=data[main_column],
                mode='lines',
                name='Normal',
                line=dict(color='blue'),
                opacity=0.7
            ),
            row=3, col=1
        )
        
        if len(anomalies) > 0:
            fig.add_trace(
                go.Scatter(
                    x=anomalies[time_column],
                    y=anomalies[main_column],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=8, symbol='x')
                ),
                row=3, col=1
            )
        
        # 5. Distribution Changes (quarterly comparison)
        data['quarter'] = data[time_column].dt.quarter
        quarters = data['quarter'].unique()
        
        for i, quarter in enumerate(quarters[-4:]):  # Last 4 quarters
            quarter_data = data[data['quarter'] == quarter][main_column]
            fig.add_trace(
                go.Histogram(
                    x=quarter_data,
                    name=f'Q{quarter}',
                    opacity=0.7,
                    nbinsx=20
                ),
                row=3, col=2
            )
        
        fig.update_layout(
            title=title,
            showlegend=True,
            height=1200,
            template="plotly_white"
        )
        
        # Generate time series insights
        insights = self._generate_timeseries_insights(data, time_column, value_columns)
        
        return {
            "figure": fig,
            "insights": insights,
            "anomalies_detected": len(anomalies),
            "trend_direction": "increasing" if data[main_column].iloc[-30:].mean() > data[main_column].iloc[-60:-30].mean() else "decreasing"
        }
    
    def create_feature_importance_analysis(self, importance_data: Dict[str, float],
                                         feature_correlations: Optional[pd.DataFrame] = None,
                                         title: str = "Feature Importance Analysis") -> Dict[str, Any]:
        """Create comprehensive feature importance visualization."""
        
        # Sort features by importance
        sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_features)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Importance Ranking', 'Importance Distribution',
                          'Top Features Correlation', 'Cumulative Importance'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # 1. Feature Importance Ranking (top 20)
        top_features = features[:20]
        top_importance = importance[:20]
        
        colors = ['darkgreen' if imp > 0.1 else 'orange' if imp > 0.05 else 'lightblue' for imp in top_importance]
        
        fig.add_trace(
            go.Bar(
                y=top_features,
                x=top_importance,
                orientation='h',
                marker_color=colors,
                text=[f"{imp:.3f}" for imp in top_importance],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Importance Distribution
        fig.add_trace(
            go.Histogram(
                x=importance,
                nbinsx=30,
                marker_color='blue',
                opacity=0.7,
                name='Importance Distribution'
            ),
            row=1, col=2
        )
        
        # 3. Top Features Correlation (if provided)
        if feature_correlations is not None:
            top_feature_names = list(top_features[:10])
            available_features = [f for f in top_feature_names if f in feature_correlations.columns]
            
            if len(available_features) > 1:
                corr_subset = feature_correlations.loc[available_features, available_features]
                
                fig.add_trace(
                    go.Heatmap(
                        z=corr_subset.values,
                        x=corr_subset.columns,
                        y=corr_subset.columns,
                        colorscale='RdBu',
                        zmid=0,
                        text=corr_subset.round(2).values,
                        texttemplate="%{text}",
                        textfont={"size": 8},
                        showscale=False
                    ),
                    row=2, col=1
                )
        
        # 4. Cumulative Importance
        cumulative_importance = np.cumsum(importance)
        
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(features) + 1)),
                y=cumulative_importance,
                mode='lines+markers',
                name='Cumulative Importance',
                line=dict(color='red', width=2)
            ),
            row=2, col=2
        )
        
        # Add 80% line
        fig.add_hline(
            y=0.8 * cumulative_importance[-1],
            line_dash="dash",
            line_color="green",
            annotation_text="80% Threshold",
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            showlegend=True,
            height=800,
            template="plotly_white"
        )
        
        # Generate feature importance insights
        insights = self._generate_feature_insights(importance_data, feature_correlations)
        
        return {
            "figure": fig,
            "insights": insights,
            "top_features": top_features[:10],
            "features_for_80_percent": len([x for x in cumulative_importance if x <= 0.8 * cumulative_importance[-1]])
        }
    
    def _generate_distribution_insights(self, data: pd.DataFrame) -> List[VisualizationInsight]:
        """Generate insights from data distribution analysis."""
        insights = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Check for skewness
        for col in numeric_cols:
            skewness = stats.skew(data[col].dropna())
            if abs(skewness) > 1:
                insights.append(VisualizationInsight(
                    id=f"skew_{col}",
                    visualization_id="distribution_analysis",
                    insight_type="skewness",
                    description=f"Column '{col}' shows significant skewness ({skewness:.2f})",
                    confidence=0.9,
                    data_points=[skewness],
                    recommendations=["Consider log transformation", "Apply normalization"],
                    timestamp=datetime.now()
                ))
        
        # Check for outliers
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data[col] < Q1 - 1.5*IQR) | (data[col] > Q3 + 1.5*IQR)]
            
            if len(outliers) > len(data) * 0.05:  # More than 5% outliers
                insights.append(VisualizationInsight(
                    id=f"outliers_{col}",
                    visualization_id="distribution_analysis",
                    insight_type="outliers",
                    description=f"Column '{col}' has {len(outliers)} potential outliers ({len(outliers)/len(data)*100:.1f}%)",
                    confidence=0.8,
                    data_points=[len(outliers)],
                    recommendations=["Investigate outlier causes", "Consider outlier removal or capping"],
                    timestamp=datetime.now()
                ))
        
        return insights
    
    def _generate_performance_insights(self, metrics: Dict[str, Any], 
                                     predictions: np.ndarray, 
                                     actual: np.ndarray) -> List[VisualizationInsight]:
        """Generate insights from model performance analysis."""
        insights = []
        
        # Check overall performance
        if 'accuracy' in metrics:
            accuracy = metrics['accuracy']
            if accuracy < 0.7:
                insights.append(VisualizationInsight(
                    id="low_accuracy",
                    visualization_id="performance_dashboard",
                    insight_type="performance",
                    description=f"Model accuracy is low ({accuracy:.3f})",
                    confidence=0.9,
                    data_points=[accuracy],
                    recommendations=["Review feature engineering", "Try different algorithms", "Increase training data"],
                    timestamp=datetime.now()
                ))
        
        # Check for overfitting indicators
        if 'train_accuracy' in metrics and 'test_accuracy' in metrics:
            train_acc = metrics['train_accuracy']
            test_acc = metrics['test_accuracy']
            if train_acc - test_acc > 0.1:
                insights.append(VisualizationInsight(
                    id="overfitting",
                    visualization_id="performance_dashboard",
                    insight_type="overfitting",
                    description=f"Potential overfitting detected (train: {train_acc:.3f}, test: {test_acc:.3f})",
                    confidence=0.8,
                    data_points=[train_acc, test_acc],
                    recommendations=["Add regularization", "Reduce model complexity", "Increase training data"],
                    timestamp=datetime.now()
                ))
        
        return insights
    
    def _generate_bias_insights(self, bias_metrics: Dict[str, float], 
                              data: pd.DataFrame, 
                              protected_attributes: List[str]) -> List[VisualizationInsight]:
        """Generate insights from bias analysis."""
        insights = []
        
        for attr in protected_attributes:
            dp_key = f"{attr}_demographic_parity"
            if dp_key in bias_metrics:
                dp_value = bias_metrics[dp_key]
                if dp_value > 0.1:  # Threshold for significant bias
                    insights.append(VisualizationInsight(
                        id=f"bias_{attr}",
                        visualization_id="bias_analysis",
                        insight_type="bias",
                        description=f"Significant bias detected for '{attr}' (demographic parity: {dp_value:.3f})",
                        confidence=0.9,
                        data_points=[dp_value],
                        recommendations=["Apply bias mitigation techniques", "Review training data", "Consider fairness constraints"],
                        timestamp=datetime.now()
                    ))
        
        return insights
    
    def _generate_lineage_insights(self, graph: nx.DiGraph, 
                                 lineage_data: Dict[str, Any]) -> List[VisualizationInsight]:
        """Generate insights from data lineage analysis."""
        insights = []
        
        # Check for circular dependencies
        if not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            insights.append(VisualizationInsight(
                id="circular_dependencies",
                visualization_id="lineage_visualization",
                insight_type="dependency",
                description=f"Circular dependencies detected in {len(cycles)} cycles",
                confidence=1.0,
                data_points=cycles,
                recommendations=["Review pipeline design", "Break circular dependencies"],
                timestamp=datetime.now()
            ))
        
        # Check for single points of failure
        critical_nodes = [node for node in graph.nodes() if graph.out_degree(node) > 5]
        if critical_nodes:
            insights.append(VisualizationInsight(
                id="critical_nodes",
                visualization_id="lineage_visualization",
                insight_type="dependency",
                description=f"Critical nodes with high fan-out detected: {critical_nodes}",
                confidence=0.8,
                data_points=critical_nodes,
                recommendations=["Consider redundancy", "Implement monitoring"],
                timestamp=datetime.now()
            ))
        
        return insights
    
    def _generate_timeseries_insights(self, data: pd.DataFrame, 
                                    time_column: str, 
                                    value_columns: List[str]) -> List[VisualizationInsight]:
        """Generate insights from time series analysis."""
        insights = []
        
        main_column = value_columns[0]
        
        # Check for trend
        recent_data = data.tail(30)[main_column]
        older_data = data.tail(60).head(30)[main_column]
        
        if recent_data.mean() > older_data.mean() * 1.1:
            insights.append(VisualizationInsight(
                id="increasing_trend",
                visualization_id="timeseries_analysis",
                insight_type="trend",
                description=f"Strong increasing trend detected in {main_column}",
                confidence=0.8,
                data_points=[recent_data.mean(), older_data.mean()],
                recommendations=["Monitor for potential issues", "Plan for capacity"],
                timestamp=datetime.now()
            ))
        
        return insights
    
    def _generate_feature_insights(self, importance_data: Dict[str, float],
                                 correlations: Optional[pd.DataFrame] = None) -> List[VisualizationInsight]:
        """Generate insights from feature importance analysis."""
        insights = []
        
        # Check for feature dominance
        sorted_importance = sorted(importance_data.values(), reverse=True)
        top_feature_importance = sorted_importance[0]
        
        if top_feature_importance > 0.5:
            top_feature = max(importance_data.items(), key=lambda x: x[1])[0]
            insights.append(VisualizationInsight(
                id="feature_dominance",
                visualization_id="feature_importance",
                insight_type="dominance",
                description=f"Feature '{top_feature}' dominates with importance {top_feature_importance:.3f}",
                confidence=0.9,
                data_points=[top_feature_importance],
                recommendations=["Check for data leakage", "Consider feature engineering"],
                timestamp=datetime.now()
            ))
        
        # Check for low-importance features
        low_importance_features = [f for f, imp in importance_data.items() if imp < 0.01]
        if len(low_importance_features) > len(importance_data) * 0.5:
            insights.append(VisualizationInsight(
                id="many_low_importance",
                visualization_id="feature_importance",
                insight_type="irrelevant_features",
                description=f"{len(low_importance_features)} features have very low importance",
                confidence=0.8,
                data_points=[len(low_importance_features)],
                recommendations=["Consider feature selection", "Remove irrelevant features"],
                timestamp=datetime.now()
            ))
        
        return insights
    
    def save_visualization_config(self, config: VisualizationConfig) -> str:
        """Save visualization configuration to database."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO visualizations 
                (id, title, type, chart_type, data_source, parameters, styling, interactive, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                config.id, config.title, config.type.value, config.chart_type.value,
                config.data_source, json.dumps(config.parameters), json.dumps(config.styling),
                config.interactive, config.created_at.isoformat(), config.updated_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            return config.id
    
    def get_visualization_insights(self, visualization_id: str) -> List[VisualizationInsight]:
        """Retrieve insights for a specific visualization."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM visualization_insights WHERE visualization_id = ?
            ORDER BY timestamp DESC
        ''', (visualization_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        insights = []
        for row in rows:
            insights.append(VisualizationInsight(
                id=row[0],
                visualization_id=row[1],
                insight_type=row[2],
                description=row[3],
                confidence=row[4],
                data_points=json.loads(row[5]),
                recommendations=json.loads(row[6]),
                timestamp=datetime.fromisoformat(row[7])
            ))
        
        return insights
    
    def create_custom_dashboard(self, visualizations: List[Dict[str, Any]], 
                              title: str = "Custom Dashboard") -> Dict[str, Any]:
        """Create a custom dashboard with multiple visualizations."""
        
        n_viz = len(visualizations)
        cols = min(3, n_viz)
        rows = (n_viz + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[viz.get('title', f'Visualization {i+1}') for i, viz in enumerate(visualizations)],
            specs=[[{} for _ in range(cols)] for _ in range(rows)]
        )
        
        for i, viz_config in enumerate(visualizations):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            # Add visualization based on type
            viz_type = viz_config.get('type', 'bar')
            data = viz_config.get('data', {})
            
            if viz_type == 'bar':
                fig.add_trace(
                    go.Bar(
                        x=data.get('x', []),
                        y=data.get('y', []),
                        name=viz_config.get('name', f'Bar {i+1}'),
                        marker_color=viz_config.get('color', 'blue')
                    ),
                    row=row, col=col
                )
            elif viz_type == 'line':
                fig.add_trace(
                    go.Scatter(
                        x=data.get('x', []),
                        y=data.get('y', []),
                        mode='lines',
                        name=viz_config.get('name', f'Line {i+1}'),
                        line=dict(color=viz_config.get('color', 'blue'))
                    ),
                    row=row, col=col
                )
            elif viz_type == 'scatter':
                fig.add_trace(
                    go.Scatter(
                        x=data.get('x', []),
                        y=data.get('y', []),
                        mode='markers',
                        name=viz_config.get('name', f'Scatter {i+1}'),
                        marker=dict(color=viz_config.get('color', 'blue'))
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title=title,
            showlegend=True,
            height=400 * rows,
            template="plotly_white"
        )
        
        return {
            "figure": fig,
            "config": {
                "visualizations": len(visualizations),
                "layout": f"{rows}x{cols}"
            }
        }

# Example usage
if __name__ == "__main__":
    # Initialize the visualization engine
    viz_engine = IntelligentVisualizationEngine()
    
    # Example data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.exponential(1, 1000),
        'feature3': np.random.uniform(-1, 1, 1000),
        'target': np.random.choice([0, 1], 1000),
        'protected_attr': np.random.choice(['A', 'B', 'C'], 1000),
        'timestamp': pd.date_range('2023-01-01', periods=1000, freq='D')
    })
    
    # Create data distribution analysis
    dist_analysis = viz_engine.create_data_distribution_analysis(
        sample_data.drop(['timestamp'], axis=1),
        "Sample Data Distribution Analysis"
    )
    
    print("Distribution Analysis completed:")
    print(f"- Total records: {dist_analysis['summary']['total_records']}")
    print(f"- Numeric features: {dist_analysis['summary']['numeric_features']}")
    print(f"- Missing data: {dist_analysis['summary']['missing_data_percentage']:.2f}%")
    print(f"- Insights generated: {len(dist_analysis['insights'])}")
    
    # Create model performance dashboard
    sample_metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88,
        'f1_score': 0.85,
        'feature_importance': {
            'feature1': 0.4,
            'feature2': 0.35,
            'feature3': 0.25
        }
    }
    
    predictions = np.random.choice([0, 1], 1000)
    actual = sample_data['target'].values
    
    perf_analysis = viz_engine.create_model_performance_dashboard(
        sample_metrics,
        predictions,
        actual,
        "Model Performance Analysis"
    )
    
    print(f"\nPerformance Analysis completed:")
    print(f"- Insights generated: {len(perf_analysis['insights'])}")
    
    # Create bias analysis
    bias_analysis = viz_engine.create_bias_analysis_visualization(
        sample_data,
        ['protected_attr'],
        'target',
        predictions,
        "Bias Analysis Dashboard"
    )
    
    print(f"\nBias Analysis completed:")
    print(f"- Bias metrics calculated: {len(bias_analysis['bias_metrics'])}")
    print(f"- Insights generated: {len(bias_analysis['insights'])}")
    
    print("\nIntelligent Visualization Engine initialized and demonstrated!")
