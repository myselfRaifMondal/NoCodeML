"""
Real-Time Analytics Dashboard Module

This module provides comprehensive real-time monitoring and visualization capabilities
for the NoCodeML platform, integrating with MLOps and Data Catalog modules to deliver
enterprise-grade analytics and reporting.

Features:
- Real-time model performance monitoring with interactive dashboards
- Data quality visualization and trend analysis
- Customizable KPI tracking and alerting
- Integration with monitoring systems for alerts and notifications
- Interactive charts and graphs for drift detection and model explainability
- Executive-level reporting and compliance dashboards
"""

import json
import sqlite3
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path
import threading
import time
from collections import defaultdict, deque
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from flask import Flask

# Import enterprise modules
from .mlops_platform import MLOpsPlatform, ModelMetrics, AlertSeverity
from .data_catalog import DataCatalog, DataQualityMetrics


class DashboardTheme(Enum):
    """Dashboard visual themes"""
    LIGHT = "light"
    DARK = "dark"
    CORPORATE = "corporate"
    SCIENTIFIC = "scientific"


class ChartType(Enum):
    """Available chart types for visualizations"""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    PIE = "pie"
    HISTOGRAM = "histogram"
    BOX = "box"
    AREA = "area"
    TREEMAP = "treemap"


class MetricCategory(Enum):
    """Categories for organizing metrics"""
    MODEL_PERFORMANCE = "model_performance"
    DATA_QUALITY = "data_quality"
    SYSTEM_HEALTH = "system_health"
    BUSINESS_KPI = "business_kpi"
    COMPLIANCE = "compliance"
    USER_ACTIVITY = "user_activity"


@dataclass
class DashboardConfig:
    """Configuration for dashboard appearance and behavior"""
    theme: DashboardTheme = DashboardTheme.CORPORATE
    refresh_interval: int = 30  # seconds
    max_data_points: int = 1000
    enable_real_time: bool = True
    enable_alerts: bool = True
    auto_layout: bool = True
    export_formats: List[str] = None
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["pdf", "png", "html", "csv"]


@dataclass
class KPIDefinition:
    """Definition for a Key Performance Indicator"""
    name: str
    category: MetricCategory
    description: str
    query: str  # SQL query or metric calculation
    target_value: Optional[float] = None
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    unit: str = ""
    chart_type: ChartType = ChartType.GAUGE
    is_higher_better: bool = True
    update_frequency: int = 60  # seconds
    enabled: bool = True


@dataclass
class WidgetConfig:
    """Configuration for dashboard widgets"""
    widget_id: str
    title: str
    widget_type: str
    position: Tuple[int, int]  # (row, col)
    size: Tuple[int, int]  # (width, height)
    data_source: str
    config: Dict[str, Any]
    refresh_rate: int = 30
    visible: bool = True


@dataclass
class AlertRule:
    """Configuration for dashboard alerts"""
    rule_id: str
    name: str
    metric_name: str
    condition: str  # e.g., ">", "<", "==", "!="
    threshold: float
    severity: AlertSeverity
    message_template: str
    enabled: bool = True
    cooldown_minutes: int = 15


class RealTimeDataStream:
    """Manages real-time data streaming for dashboard updates"""
    
    def __init__(self, max_buffer_size: int = 10000):
        self.max_buffer_size = max_buffer_size
        self.data_buffers = defaultdict(lambda: deque(maxlen=max_buffer_size))
        self.subscribers = defaultdict(list)
        self.running = False
        self.update_thread = None
        
    def add_data_point(self, stream_name: str, timestamp: datetime, value: Any):
        """Add a data point to the stream"""
        self.data_buffers[stream_name].append({
            'timestamp': timestamp,
            'value': value
        })
        self._notify_subscribers(stream_name)
    
    def subscribe(self, stream_name: str, callback):
        """Subscribe to updates for a specific data stream"""
        self.subscribers[stream_name].append(callback)
    
    def _notify_subscribers(self, stream_name: str):
        """Notify all subscribers of updates"""
        for callback in self.subscribers[stream_name]:
            try:
                callback(stream_name, list(self.data_buffers[stream_name]))
            except Exception as e:
                logging.error(f"Error notifying subscriber: {e}")
    
    def get_recent_data(self, stream_name: str, limit: int = 100) -> List[Dict]:
        """Get recent data points from a stream"""
        data = list(self.data_buffers[stream_name])
        return data[-limit:] if limit else data
    
    def start(self):
        """Start the real-time data streaming"""
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def stop(self):
        """Stop the real-time data streaming"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
    
    def _update_loop(self):
        """Main update loop for real-time streaming"""
        while self.running:
            time.sleep(1)  # Update every second


class AnalyticsDashboard:
    """
    Real-time analytics dashboard for NoCodeML platform
    
    Provides comprehensive monitoring and visualization capabilities including:
    - Model performance tracking
    - Data quality monitoring
    - System health dashboards
    - Custom KPI tracking
    - Alert management
    """
    
    def __init__(self, 
                 db_path: str = "dashboard.db",
                 mlops_platform: Optional[MLOpsPlatform] = None,
                 data_catalog: Optional[DataCatalog] = None,
                 config: Optional[DashboardConfig] = None):
        self.db_path = db_path
        self.mlops = mlops_platform
        self.data_catalog = data_catalog
        self.config = config or DashboardConfig()
        
        # Initialize components
        self.data_stream = RealTimeDataStream()
        self.kpis = {}
        self.widgets = {}
        self.alert_rules = {}
        self.dash_app = None
        
        # Setup database
        self._init_database()
        
        # Load configurations
        self._load_default_kpis()
        self._load_default_widgets()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Start real-time updates if enabled
        if self.config.enable_real_time:
            self.data_stream.start()
            self._start_data_collection()
    
    def _init_database(self):
        """Initialize dashboard database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS dashboard_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS kpi_definitions (
                    name TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    description TEXT,
                    query TEXT NOT NULL,
                    target_value REAL,
                    warning_threshold REAL,
                    critical_threshold REAL,
                    unit TEXT,
                    chart_type TEXT,
                    is_higher_better BOOLEAN,
                    update_frequency INTEGER,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS widget_configs (
                    widget_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    widget_type TEXT NOT NULL,
                    position_row INTEGER,
                    position_col INTEGER,
                    size_width INTEGER,
                    size_height INTEGER,
                    data_source TEXT,
                    config TEXT,
                    refresh_rate INTEGER DEFAULT 30,
                    visible BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS alert_rules (
                    rule_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    severity TEXT NOT NULL,
                    message_template TEXT,
                    enabled BOOLEAN DEFAULT TRUE,
                    cooldown_minutes INTEGER DEFAULT 15,
                    last_triggered DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS dashboard_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    page_views INTEGER DEFAULT 0,
                    interactions INTEGER DEFAULT 0
                );
                
                CREATE TABLE IF NOT EXISTS export_history (
                    export_id TEXT PRIMARY KEY,
                    dashboard_name TEXT,
                    format TEXT,
                    file_path TEXT,
                    exported_by TEXT,
                    export_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    file_size INTEGER
                );
                
                CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON dashboard_metrics(metric_name, timestamp);
                CREATE INDEX IF NOT EXISTS idx_alert_rules_metric ON alert_rules(metric_name);
            """)
    
    def _load_default_kpis(self):
        """Load default KPI definitions"""
        default_kpis = [
            KPIDefinition(
                name="model_accuracy",
                category=MetricCategory.MODEL_PERFORMANCE,
                description="Overall model accuracy across all deployed models",
                query="SELECT AVG(accuracy) FROM model_metrics WHERE timestamp > datetime('now', '-1 hour')",
                target_value=0.9,
                warning_threshold=0.8,
                critical_threshold=0.7,
                unit="%",
                chart_type=ChartType.GAUGE
            ),
            KPIDefinition(
                name="data_quality_score",
                category=MetricCategory.DATA_QUALITY,
                description="Average data quality score across all datasets",
                query="SELECT AVG(quality_score) FROM data_quality WHERE timestamp > datetime('now', '-1 hour')",
                target_value=0.95,
                warning_threshold=0.85,
                critical_threshold=0.75,
                unit="score",
                chart_type=ChartType.GAUGE
            ),
            KPIDefinition(
                name="prediction_latency",
                category=MetricCategory.SYSTEM_HEALTH,
                description="Average prediction latency across all models",
                query="SELECT AVG(latency_ms) FROM prediction_logs WHERE timestamp > datetime('now', '-1 hour')",
                target_value=100,
                warning_threshold=200,
                critical_threshold=500,
                unit="ms",
                chart_type=ChartType.LINE,
                is_higher_better=False
            ),
            KPIDefinition(
                name="data_drift_score",
                category=MetricCategory.DATA_QUALITY,
                description="Maximum data drift score across all monitored models",
                query="SELECT MAX(drift_score) FROM drift_detection WHERE timestamp > datetime('now', '-1 hour')",
                target_value=0.1,
                warning_threshold=0.3,
                critical_threshold=0.5,
                unit="score",
                chart_type=ChartType.LINE,
                is_higher_better=False
            ),
            KPIDefinition(
                name="active_models",
                category=MetricCategory.SYSTEM_HEALTH,
                description="Number of currently active deployed models",
                query="SELECT COUNT(*) FROM deployments WHERE status = 'active'",
                target_value=None,
                unit="models",
                chart_type=ChartType.GAUGE
            )
        ]
        
        for kpi in default_kpis:
            self.add_kpi(kpi)
    
    def _load_default_widgets(self):
        """Load default widget configurations"""
        default_widgets = [
            WidgetConfig(
                widget_id="model_performance_overview",
                title="Model Performance Overview",
                widget_type="metrics_grid",
                position=(0, 0),
                size=(6, 4),
                data_source="model_metrics",
                config={"metrics": ["accuracy", "precision", "recall", "f1_score"]}
            ),
            WidgetConfig(
                widget_id="data_quality_trends",
                title="Data Quality Trends",
                widget_type="line_chart",
                position=(0, 6),
                size=(6, 4),
                data_source="data_quality",
                config={"y_axis": "quality_score", "time_range": "24h"}
            ),
            WidgetConfig(
                widget_id="system_alerts",
                title="Recent Alerts",
                widget_type="alert_list",
                position=(4, 0),
                size=(12, 3),
                data_source="alerts",
                config={"max_items": 10, "severity_filter": ["warning", "critical"]}
            ),
            WidgetConfig(
                widget_id="drift_detection",
                title="Data Drift Detection",
                widget_type="heatmap",
                position=(7, 0),
                size=(6, 4),
                data_source="drift_metrics",
                config={"features": "auto", "threshold": 0.3}
            ),
            WidgetConfig(
                widget_id="prediction_volume",
                title="Prediction Volume",
                widget_type="area_chart",
                position=(7, 6),
                size=(6, 4),
                data_source="prediction_logs",
                config={"aggregation": "count", "interval": "5m"}
            )
        ]
        
        for widget in default_widgets:
            self.add_widget(widget)
    
    def add_kpi(self, kpi: KPIDefinition):
        """Add a new KPI definition"""
        self.kpis[kpi.name] = kpi
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO kpi_definitions 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                kpi.name, kpi.category.value, kpi.description, kpi.query,
                kpi.target_value, kpi.warning_threshold, kpi.critical_threshold,
                kpi.unit, kpi.chart_type.value, kpi.is_higher_better,
                kpi.update_frequency, kpi.enabled
            ))
    
    def add_widget(self, widget: WidgetConfig):
        """Add a new widget configuration"""
        self.widgets[widget.widget_id] = widget
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO widget_configs 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                widget.widget_id, widget.title, widget.widget_type,
                widget.position[0], widget.position[1],
                widget.size[0], widget.size[1],
                widget.data_source, json.dumps(widget.config),
                widget.refresh_rate, widget.visible
            ))
    
    def add_alert_rule(self, alert_rule: AlertRule):
        """Add a new alert rule"""
        self.alert_rules[alert_rule.rule_id] = alert_rule
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alert_rules 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, CURRENT_TIMESTAMP)
            """, (
                alert_rule.rule_id, alert_rule.name, alert_rule.metric_name,
                alert_rule.condition, alert_rule.threshold, alert_rule.severity.value,
                alert_rule.message_template, alert_rule.enabled,
                alert_rule.cooldown_minutes
            ))
    
    def record_metric(self, metric_name: str, category: MetricCategory, 
                     value: float, metadata: Optional[Dict] = None):
        """Record a new metric value"""
        timestamp = datetime.utcnow()
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO dashboard_metrics (metric_name, category, value, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (metric_name, category.value, value, timestamp, 
                  json.dumps(metadata) if metadata else None))
        
        # Add to real-time stream
        self.data_stream.add_data_point(metric_name, timestamp, value)
        
        # Check alert rules
        self._check_alert_rules(metric_name, value)
    
    def _check_alert_rules(self, metric_name: str, value: float):
        """Check if any alert rules are triggered"""
        for rule in self.alert_rules.values():
            if rule.metric_name == metric_name and rule.enabled:
                triggered = False
                
                if rule.condition == ">" and value > rule.threshold:
                    triggered = True
                elif rule.condition == "<" and value < rule.threshold:
                    triggered = True
                elif rule.condition == "==" and value == rule.threshold:
                    triggered = True
                elif rule.condition == "!=" and value != rule.threshold:
                    triggered = True
                
                if triggered:
                    self._trigger_alert(rule, value)
    
    def _trigger_alert(self, rule: AlertRule, value: float):
        """Trigger an alert"""
        # Check cooldown period
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT last_triggered FROM alert_rules 
                WHERE rule_id = ?
            """, (rule.rule_id,))
            result = cursor.fetchone()
            
            if result and result[0]:
                last_triggered = datetime.fromisoformat(result[0])
                if (datetime.utcnow() - last_triggered).total_seconds() < rule.cooldown_minutes * 60:
                    return  # Still in cooldown
        
        # Create alert message
        message = rule.message_template.format(
            metric_name=rule.metric_name,
            value=value,
            threshold=rule.threshold,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Log alert
        self.logger.warning(f"Alert triggered: {rule.name} - {message}")
        
        # Update last triggered time
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE alert_rules 
                SET last_triggered = CURRENT_TIMESTAMP 
                WHERE rule_id = ?
            """, (rule.rule_id,))
        
        # If MLOps platform is available, use its alerting system
        if self.mlops:
            self.mlops.create_alert(
                model_id="dashboard",
                alert_type="kpi_threshold",
                severity=rule.severity,
                message=message,
                metadata={"rule_id": rule.rule_id, "value": value}
            )
    
    def get_kpi_value(self, kpi_name: str) -> Optional[float]:
        """Get current value for a KPI"""
        if kpi_name not in self.kpis:
            return None
        
        kpi = self.kpis[kpi_name]
        
        try:
            # Execute KPI query
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(kpi.query)
                result = cursor.fetchone()
                return result[0] if result and result[0] is not None else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating KPI {kpi_name}: {e}")
            return None
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> pd.DataFrame:
        """Get historical data for a metric"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT timestamp, value, metadata 
                FROM dashboard_metrics 
                WHERE metric_name = ? AND timestamp > datetime('now', '-{} hours')
                ORDER BY timestamp
            """.format(hours)
            
            return pd.read_sql_query(query, conn, params=(metric_name,))
    
    def create_visualization(self, metric_name: str, chart_type: ChartType, 
                           hours: int = 24, **kwargs) -> go.Figure:
        """Create a visualization for a metric"""
        df = self.get_metric_history(metric_name, hours)
        
        if df.empty:
            # Return empty chart
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for {metric_name}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if chart_type == ChartType.LINE:
            fig = px.line(df, x='timestamp', y='value', 
                         title=f"{metric_name} Over Time")
        elif chart_type == ChartType.BAR:
            fig = px.bar(df, x='timestamp', y='value', 
                        title=f"{metric_name} Values")
        elif chart_type == ChartType.SCATTER:
            fig = px.scatter(df, x='timestamp', y='value', 
                           title=f"{metric_name} Scatter Plot")
        elif chart_type == ChartType.AREA:
            fig = px.area(df, x='timestamp', y='value', 
                         title=f"{metric_name} Area Chart")
        elif chart_type == ChartType.GAUGE:
            current_value = df['value'].iloc[-1] if not df.empty else 0
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=current_value,
                title={'text': metric_name},
                gauge={'axis': {'range': [None, current_value * 1.2]},
                      'threshold': {'line': {'color': "red", 'width': 4},
                                  'thickness': 0.75, 'value': current_value * 0.9}}
            ))
        else:
            # Default to line chart
            fig = px.line(df, x='timestamp', y='value', 
                         title=f"{metric_name} Over Time")
        
        # Apply theme
        if self.config.theme == DashboardTheme.DARK:
            fig.update_layout(template="plotly_dark")
        elif self.config.theme == DashboardTheme.CORPORATE:
            fig.update_layout(template="simple_white")
        
        return fig
    
    def _start_data_collection(self):
        """Start collecting data from integrated systems"""
        def collect_data():
            while self.config.enable_real_time:
                try:
                    # Collect MLOps metrics
                    if self.mlops:
                        self._collect_mlops_metrics()
                    
                    # Collect data catalog metrics
                    if self.data_catalog:
                        self._collect_data_catalog_metrics()
                    
                    # Update KPIs
                    self._update_kpis()
                    
                except Exception as e:
                    self.logger.error(f"Error in data collection: {e}")
                
                time.sleep(self.config.refresh_interval)
        
        collection_thread = threading.Thread(target=collect_data)
        collection_thread.daemon = True
        collection_thread.start()
    
    def _collect_mlops_metrics(self):
        """Collect metrics from MLOps platform"""
        try:
            # Get active deployments
            with sqlite3.connect(self.mlops.db_path) as conn:
                cursor = conn.execute("""
                    SELECT model_id, COUNT(*) as prediction_count,
                           AVG(CASE WHEN json_extract(metrics, '$.accuracy') IS NOT NULL 
                               THEN json_extract(metrics, '$.accuracy') END) as avg_accuracy,
                           AVG(CASE WHEN json_extract(metrics, '$.latency_ms') IS NOT NULL 
                               THEN json_extract(metrics, '$.latency_ms') END) as avg_latency
                    FROM monitoring_data 
                    WHERE timestamp > datetime('now', '-1 hour')
                    GROUP BY model_id
                """)
                
                for row in cursor.fetchall():
                    model_id, pred_count, accuracy, latency = row
                    
                    if accuracy:
                        self.record_metric(
                            f"model_accuracy_{model_id}",
                            MetricCategory.MODEL_PERFORMANCE,
                            accuracy
                        )
                    
                    if latency:
                        self.record_metric(
                            f"prediction_latency_{model_id}",
                            MetricCategory.SYSTEM_HEALTH,
                            latency
                        )
                    
                    self.record_metric(
                        f"prediction_volume_{model_id}",
                        MetricCategory.SYSTEM_HEALTH,
                        pred_count or 0
                    )
        except Exception as e:
            self.logger.error(f"Error collecting MLOps metrics: {e}")
    
    def _collect_data_catalog_metrics(self):
        """Collect metrics from data catalog"""
        try:
            with sqlite3.connect(self.data_catalog.db_path) as conn:
                # Get average data quality score
                cursor = conn.execute("""
                    SELECT AVG(quality_score) as avg_quality
                    FROM datasets 
                    WHERE updated_at > datetime('now', '-1 hour')
                """)
                result = cursor.fetchone()
                if result and result[0]:
                    self.record_metric(
                        "data_quality_score",
                        MetricCategory.DATA_QUALITY,
                        result[0]
                    )
                
                # Get PII detection stats
                cursor = conn.execute("""
                    SELECT 
                        SUM(CASE WHEN has_pii = 1 THEN 1 ELSE 0 END) as pii_datasets,
                        COUNT(*) as total_datasets
                    FROM datasets
                """)
                result = cursor.fetchone()
                if result:
                    pii_ratio = result[0] / result[1] if result[1] > 0 else 0
                    self.record_metric(
                        "pii_ratio",
                        MetricCategory.COMPLIANCE,
                        pii_ratio
                    )
        except Exception as e:
            self.logger.error(f"Error collecting data catalog metrics: {e}")
    
    def _update_kpis(self):
        """Update all KPI values"""
        for kpi_name, kpi in self.kpis.items():
            if kpi.enabled:
                value = self.get_kpi_value(kpi_name)
                if value is not None:
                    self.record_metric(kpi_name, kpi.category, value)
    
    def create_dash_app(self, debug: bool = False) -> dash.Dash:
        """Create and configure Dash application"""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Define layout
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("NoCodeML Analytics Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # KPI Cards Row
            dbc.Row([
                dbc.Col([
                    self._create_kpi_card(kpi_name) 
                    for kpi_name in list(self.kpis.keys())[:4]
                ], width=12)
            ], className="mb-4"),
            
            # Charts Row
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="model-performance-chart")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="data-quality-chart")
                ], width=6)
            ], className="mb-4"),
            
            # Alerts and System Status
            dbc.Row([
                dbc.Col([
                    html.H4("Recent Alerts"),
                    html.Div(id="alerts-list")
                ], width=6),
                dbc.Col([
                    html.H4("System Status"),
                    html.Div(id="system-status")
                ], width=6)
            ]),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=self.config.refresh_interval * 1000,  # milliseconds
                n_intervals=0
            )
        ], fluid=True)
        
        # Setup callbacks
        self._setup_callbacks(app)
        
        self.dash_app = app
        return app
    
    def _create_kpi_card(self, kpi_name: str) -> dbc.Card:
        """Create a KPI card component"""
        kpi = self.kpis.get(kpi_name)
        if not kpi:
            return dbc.Card("KPI not found")
        
        value = self.get_kpi_value(kpi_name) or 0
        
        # Determine card color based on thresholds
        color = "success"
        if kpi.critical_threshold and (
            (kpi.is_higher_better and value < kpi.critical_threshold) or
            (not kpi.is_higher_better and value > kpi.critical_threshold)
        ):
            color = "danger"
        elif kpi.warning_threshold and (
            (kpi.is_higher_better and value < kpi.warning_threshold) or
            (not kpi.is_higher_better and value > kpi.warning_threshold)
        ):
            color = "warning"
        
        return dbc.Card([
            dbc.CardBody([
                html.H4(f"{value:.2f} {kpi.unit}", className="card-title"),
                html.P(kpi.name.replace('_', ' ').title(), className="card-text")
            ])
        ], color=color, outline=True, className="mb-2")
    
    def _setup_callbacks(self, app: dash.Dash):
        """Setup Dash callbacks for interactivity"""
        
        @app.callback(
            [Output('model-performance-chart', 'figure'),
             Output('data-quality-chart', 'figure'),
             Output('alerts-list', 'children'),
             Output('system-status', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            # Update model performance chart
            model_perf_fig = self.create_visualization(
                "model_accuracy", ChartType.LINE, hours=24
            )
            
            # Update data quality chart
            data_quality_fig = self.create_visualization(
                "data_quality_score", ChartType.LINE, hours=24
            )
            
            # Get recent alerts
            alerts_list = self._get_recent_alerts_component()
            
            # Get system status
            system_status = self._get_system_status_component()
            
            return model_perf_fig, data_quality_fig, alerts_list, system_status
    
    def _get_recent_alerts_component(self) -> html.Div:
        """Get recent alerts as HTML component"""
        if not self.mlops:
            return html.P("No alerts available")
        
        try:
            with sqlite3.connect(self.mlops.db_path) as conn:
                cursor = conn.execute("""
                    SELECT alert_type, severity, message, created_at
                    FROM alerts 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """)
                alerts = cursor.fetchall()
            
            if not alerts:
                return html.P("No recent alerts")
            
            alert_items = []
            for alert_type, severity, message, created_at in alerts:
                color = "danger" if severity == "critical" else "warning"
                alert_items.append(
                    dbc.Alert([
                        html.Strong(f"{alert_type}: "),
                        message,
                        html.Small(f" ({created_at})", className="text-muted")
                    ], color=color, className="mb-2")
                )
            
            return html.Div(alert_items)
            
        except Exception as e:
            return html.P(f"Error loading alerts: {e}")
    
    def _get_system_status_component(self) -> html.Div:
        """Get system status as HTML component"""
        status_items = []
        
        # Check MLOps platform status
        if self.mlops:
            try:
                with sqlite3.connect(self.mlops.db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM deployments WHERE status = 'active'")
                    active_models = cursor.fetchone()[0]
                    
                    status_items.append(
                        html.P([
                            html.Strong("Active Models: "),
                            html.Span(str(active_models), className="badge badge-success")
                        ])
                    )
            except Exception:
                status_items.append(html.P("MLOps status: Unknown"))
        
        # Check data catalog status
        if self.data_catalog:
            try:
                with sqlite3.connect(self.data_catalog.db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM datasets")
                    total_datasets = cursor.fetchone()[0]
                    
                    status_items.append(
                        html.P([
                            html.Strong("Cataloged Datasets: "),
                            html.Span(str(total_datasets), className="badge badge-info")
                        ])
                    )
            except Exception:
                status_items.append(html.P("Data Catalog status: Unknown"))
        
        # Add dashboard uptime
        status_items.append(
            html.P([
                html.Strong("Dashboard Status: "),
                html.Span("Online", className="badge badge-success")
            ])
        )
        
        return html.Div(status_items)
    
    def export_dashboard(self, format_type: str = "pdf", 
                        dashboard_name: str = "dashboard") -> str:
        """Export dashboard to various formats"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{dashboard_name}_{timestamp}.{format_type}"
        
        if format_type == "pdf":
            # Implementation for PDF export would go here
            # This would typically use libraries like weasyprint or reportlab
            pass
        elif format_type == "png":
            # Implementation for PNG export would go here
            # This would capture dashboard screenshots
            pass
        elif format_type == "html":
            # Export as standalone HTML
            if self.dash_app:
                # Generate static HTML version
                pass
        elif format_type == "csv":
            # Export metrics data as CSV
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("""
                    SELECT * FROM dashboard_metrics 
                    WHERE timestamp > datetime('now', '-24 hours')
                """, conn)
                df.to_csv(filename, index=False)
        
        # Log export
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO export_history (export_id, dashboard_name, format, file_path, export_time)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (f"export_{timestamp}", dashboard_name, format_type, filename))
        
        return filename
    
    def get_dashboard_analytics(self) -> Dict[str, Any]:
        """Get analytics about dashboard usage"""
        with sqlite3.connect(self.db_path) as conn:
            # Get session statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_sessions,
                    AVG(page_views) as avg_page_views,
                    AVG(interactions) as avg_interactions,
                    MAX(start_time) as last_access
                FROM dashboard_sessions
                WHERE start_time > datetime('now', '-30 days')
            """)
            session_stats = cursor.fetchone()
            
            # Get export statistics
            cursor = conn.execute("""
                SELECT format, COUNT(*) as count
                FROM export_history
                WHERE export_time > datetime('now', '-30 days')
                GROUP BY format
            """)
            export_stats = dict(cursor.fetchall())
            
            # Get most viewed metrics
            cursor = conn.execute("""
                SELECT metric_name, COUNT(*) as views
                FROM dashboard_metrics
                WHERE timestamp > datetime('now', '-30 days')
                GROUP BY metric_name
                ORDER BY views DESC
                LIMIT 10
            """)
            popular_metrics = cursor.fetchall()
        
        return {
            "session_stats": {
                "total_sessions": session_stats[0] or 0,
                "avg_page_views": session_stats[1] or 0,
                "avg_interactions": session_stats[2] or 0,
                "last_access": session_stats[3]
            },
            "export_stats": export_stats,
            "popular_metrics": popular_metrics
        }
    
    def run_dashboard(self, host: str = "127.0.0.1", port: int = 8050, 
                     debug: bool = False):
        """Run the dashboard server"""
        if not self.dash_app:
            self.create_dash_app(debug=debug)
        
        self.logger.info(f"Starting dashboard server on {host}:{port}")
        self.dash_app.run_server(host=host, port=port, debug=debug)
    
    def stop(self):
        """Stop the dashboard and cleanup resources"""
        self.data_stream.stop()
        self.logger.info("Analytics dashboard stopped")


# Example usage and testing
if __name__ == "__main__":
    # Initialize dashboard with sample configuration
    config = DashboardConfig(
        theme=DashboardTheme.CORPORATE,
        refresh_interval=30,
        enable_real_time=True,
        enable_alerts=True
    )
    
    dashboard = AnalyticsDashboard(
        db_path="test_dashboard.db",
        config=config
    )
    
    # Add some sample data
    dashboard.record_metric("test_accuracy", MetricCategory.MODEL_PERFORMANCE, 0.92)
    dashboard.record_metric("test_latency", MetricCategory.SYSTEM_HEALTH, 150.5)
    dashboard.record_metric("data_quality", MetricCategory.DATA_QUALITY, 0.88)
    
    # Add custom alert rule
    alert_rule = AlertRule(
        rule_id="accuracy_drop",
        name="Model Accuracy Drop",
        metric_name="test_accuracy",
        condition="<",
        threshold=0.85,
        severity=AlertSeverity.WARNING,
        message_template="Model accuracy dropped to {value:.2f}, below threshold of {threshold}"
    )
    dashboard.add_alert_rule(alert_rule)
    
    # Create and run dashboard
    dashboard.create_dash_app()
    print("Dashboard created successfully!")
    print("Run dashboard.run_dashboard() to start the server")
