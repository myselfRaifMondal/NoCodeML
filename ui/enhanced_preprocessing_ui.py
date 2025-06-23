"""
Enhanced Streamlit UI for Advanced Data Preprocessing and Iterative Model Improvement

This module provides an advanced user interface for the enhanced preprocessing
and iterative improvement capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.integration.preprocessing_trainer_integration import EnhancedMLPipeline
from core.preprocessing.enhanced_preprocessor import EnhancedDataPreprocessor
from backend.models.schemas import ProblemType

class EnhancedPreprocessingUI:
    """Enhanced UI for data preprocessing with iterative improvement"""
    
    def __init__(self):
        self.pipeline = EnhancedMLPipeline(learning_enabled=True)
        
    def render_preprocessing_page(self):
        """Render the enhanced preprocessing page"""
        st.markdown("## 🧹 Enhanced Data Preprocessing & Quality Improvement")
        
        # Check if dataset is available
        if 'analysis_results' not in st.session_state or st.session_state.analysis_results is None:
            st.warning("⚠️ No dataset available. Please upload and analyze a dataset first.")
            return
        
        analysis = st.session_state.analysis_results
        df = analysis.get('dataframe')
        
        if df is None:
            st.error("❌ Dataset not available for preprocessing.")
            return
        
        # Main preprocessing interface
        self._render_preprocessing_interface(df, analysis)
    
    def _render_preprocessing_interface(self, df: pd.DataFrame, analysis: Dict[str, Any]):
        """Render the main preprocessing interface"""
        
        # Preprocessing configuration
        st.markdown("### ⚙️ Preprocessing Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_column = st.selectbox(
                "Select Target Column",
                df.columns.tolist(),
                help="Choose the column you want to predict"
            )
        
        with col2:
            problem_type = st.selectbox(
                "Problem Type",
                ["classification", "regression"],
                help="Type of machine learning problem"
            )
            problem_type = ProblemType.CLASSIFICATION if problem_type == "classification" else ProblemType.REGRESSION
        
        with col3:
            max_iterations = st.slider(
                "Max Iterations",
                min_value=1,
                max_value=5,
                value=3,
                help="Maximum number of improvement iterations"
            )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                improvement_threshold = st.slider(
                    "Improvement Threshold",
                    min_value=0.001,
                    max_value=0.1,
                    value=0.01,
                    step=0.001,
                    help="Minimum improvement required to continue iterations"
                )
                
                enable_learning = st.checkbox(
                    "Enable Learning",
                    value=True,
                    help="Learn from preprocessing results to improve future runs"
                )
            
            with col2:
                user_feedback = st.checkbox(
                    "Collect User Feedback",
                    value=True,
                    help="Collect feedback to improve preprocessing strategies"
                )
                
                detailed_logging = st.checkbox(
                    "Detailed Logging",
                    value=False,
                    help="Show detailed preprocessing logs"
                )
        
        # Data quality overview
        self._render_data_quality_overview(df)
        
        # Start preprocessing button
        if st.button("🚀 Start Enhanced Preprocessing", type="primary"):
            self._run_enhanced_preprocessing(
                df, target_column, problem_type, max_iterations, 
                improvement_threshold, enable_learning, user_feedback, detailed_logging
            )
    
    def _render_data_quality_overview(self, df: pd.DataFrame):
        """Render data quality overview"""
        st.markdown("### 📊 Current Data Quality Overview")
        
        # Quick quality metrics
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(df)) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        with col2:
            st.metric("Duplicate Rows", f"{duplicate_count} ({duplicate_pct:.1f}%)")
        
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Features", len(numeric_cols))
        
        with col4:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            st.metric("Categorical Features", len(categorical_cols))
        
        # Column-wise quality details
        with st.expander("🔍 Column-wise Quality Details"):
            quality_data = []
            
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                unique_count = df[col].nunique()
                data_type = str(df[col].dtype)
                
                quality_data.append({
                    'Column': col,
                    'Data Type': data_type,
                    'Missing Count': missing_count,
                    'Missing %': f"{missing_pct:.1f}%",
                    'Unique Values': unique_count,
                    'Cardinality Ratio': f"{unique_count/len(df):.3f}"
                })
            
            quality_df = pd.DataFrame(quality_data)
            st.dataframe(quality_df, use_container_width=True)
    
    def _run_enhanced_preprocessing(self, df: pd.DataFrame, target_column: str, 
                                  problem_type: ProblemType, max_iterations: int,
                                  improvement_threshold: float, enable_learning: bool,
                                  user_feedback: bool, detailed_logging: bool):
        """Run the enhanced preprocessing pipeline"""
        
        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.empty()
        
        # Initialize results containers
        results_container = st.empty()
        
        # Progress callback
        async def progress_callback(message: str, progress: int):
            progress_bar.progress(progress / 100)
            status_text.text(f"🔄 {message}")
            if detailed_logging:
                with log_container.container():
                    st.write(f"**{datetime.now().strftime('%H:%M:%S')}** - {message}")
        
        # User feedback callback
        async def feedback_callback(results: Dict[str, Any]) -> Dict[str, Any]:
            if not user_feedback:
                return {}
            
            # Create feedback form
            with st.form("preprocessing_feedback"):
                st.markdown("### 📝 Preprocessing Feedback")
                st.write("Please rate the preprocessing results:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    overall_satisfaction = st.slider(
                        "Overall Satisfaction",
                        min_value=1,
                        max_value=10,
                        value=7,
                        help="How satisfied are you with the overall preprocessing?"
                    )
                    
                    preprocessing_satisfaction = st.slider(
                        "Data Quality Improvement",
                        min_value=1,
                        max_value=10,
                        value=7,
                        help="How well did preprocessing improve data quality?"
                    )
                
                with col2:
                    model_satisfaction = st.slider(
                        "Feature Engineering Quality",
                        min_value=1,
                        max_value=10,
                        value=7,
                        help="How satisfied are you with feature engineering?"
                    )
                    
                    ease_of_use = st.slider(
                        "Ease of Use",
                        min_value=1,
                        max_value=10,
                        value=8,
                        help="How easy was it to use this preprocessing tool?"
                    )
                
                comments = st.text_area(
                    "Additional Comments",
                    placeholder="Any suggestions or observations about the preprocessing results..."
                )
                
                submit_feedback = st.form_submit_button("Submit Feedback")
                
                if submit_feedback:
                    return {
                        'overall_satisfaction': overall_satisfaction,
                        'preprocessing_satisfaction': preprocessing_satisfaction,
                        'model_satisfaction': model_satisfaction,
                        'ease_of_use': ease_of_use,
                        'comments': comments,
                        'timestamp': datetime.now().isoformat()
                    }
            
            return {}
        
        # Run the pipeline
        try:
            # Use asyncio to run the async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            results = loop.run_until_complete(
                self.pipeline.train_with_iterative_improvement(
                    df=df,
                    target_column=target_column,
                    problem_type=problem_type,
                    max_preprocessing_iterations=max_iterations,
                    max_training_iterations=2,  # Limit model training for UI responsiveness
                    improvement_threshold=improvement_threshold,
                    user_feedback_callback=feedback_callback if user_feedback else None,
                    progress_callback=progress_callback
                )
            )
            
            # Store results
            st.session_state.preprocessing_results = results
            
            # Display results
            self._display_preprocessing_results(results)
            
        except Exception as e:
            st.error(f"❌ Preprocessing failed: {str(e)}")
            st.exception(e)
        
        finally:
            progress_bar.progress(100)
            status_text.text("✅ Preprocessing completed!")
    
    def _display_preprocessing_results(self, results: Dict[str, Any]):
        """Display comprehensive preprocessing results"""
        st.markdown("## 📈 Preprocessing Results")
        
        # Executive summary
        self._render_executive_summary(results)
        
        # Detailed results tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Quality Metrics", 
            "🔧 Transformations", 
            "🤖 Model Performance",
            "💡 Recommendations",
            "📋 Detailed Report"
        ])
        
        with tab1:
            self._render_quality_metrics(results)
        
        with tab2:
            self._render_transformations(results)
        
        with tab3:
            self._render_model_performance(results)
        
        with tab4:
            self._render_recommendations(results)
        
        with tab5:
            self._render_detailed_report(results)
    
    def _render_executive_summary(self, results: Dict[str, Any]):
        """Render executive summary of results"""
        st.markdown("### 📋 Executive Summary")
        
        summary = results.get('pipeline_summary', {})
        preprocessing_summary = results.get('preprocessing_summary', {})
        model_summary = results.get('model_training_summary', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Processing Time",
                f"{summary.get('total_processing_time', 0):.1f}s"
            )
        
        with col2:
            st.metric(
                "Quality Improvement",
                f"{preprocessing_summary.get('total_quality_improvement', 0):.1f}%"
            )
        
        with col3:
            st.metric(
                "Issues Fixed",
                preprocessing_summary.get('issues_fixed', 0)
            )
        
        with col4:
            best_model_metrics = model_summary.get('best_model_performance', {})
            primary_metric = next(iter(best_model_metrics.values()), 0) if best_model_metrics else 0
            st.metric(
                "Model Performance",
                f"{primary_metric:.3f}"
            )
        
        # Data shape change
        original_shape = summary.get('original_data_shape', (0, 0))
        processed_shape = summary.get('processed_data_shape', (0, 0))
        
        st.info(f"📊 **Data Shape Change:** {original_shape} → {processed_shape}")
        
        if model_summary.get('best_model_name'):
            st.success(f"🏆 **Best Model:** {model_summary['best_model_name']}")
    
    def _render_quality_metrics(self, results: Dict[str, Any]):
        """Render quality metrics visualization"""
        st.markdown("### 📊 Data Quality Metrics")
        
        preprocessing_results = results.get('detailed_results', {}).get('preprocessing_results', [])
        
        if not preprocessing_results:
            st.warning("No preprocessing metrics available.")
            return
        
        # Quality improvement over iterations
        iterations = list(range(1, len(preprocessing_results) + 1))
        quality_scores = [r['metrics']['data_quality_score'] for r in preprocessing_results]
        missing_ratios = [r['metrics']['missing_data_ratio'] * 100 for r in preprocessing_results]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Quality Score Over Iterations', 'Missing Data Reduction', 
                          'Processing Time', 'Feature Quality Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Quality score trend
        fig.add_trace(
            go.Scatter(x=iterations, y=quality_scores, mode='lines+markers',
                      name='Quality Score', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Missing data reduction
        fig.add_trace(
            go.Scatter(x=iterations, y=missing_ratios, mode='lines+markers',
                      name='Missing Data %', line=dict(color='red')),
            row=1, col=2
        )
        
        # Processing time
        processing_times = [r['metrics']['preprocessing_time'] for r in preprocessing_results]
        fig.add_trace(
            go.Bar(x=iterations, y=processing_times, name='Processing Time'),
            row=2, col=1
        )
        
        # Feature quality scores
        feature_scores = [r['metrics'].get('feature_quality_score', 0) for r in preprocessing_results]
        fig.add_trace(
            go.Scatter(x=iterations, y=feature_scores, mode='lines+markers',
                      name='Feature Quality', line=dict(color='green')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Final metrics table
        st.markdown("#### Final Quality Metrics")
        final_metrics = preprocessing_results[-1]['metrics']
        
        metrics_data = {
            'Metric': ['Data Quality Score', 'Missing Data Ratio', 'Duplicate Ratio', 'Feature Quality Score'],
            'Value': [
                f"{final_metrics['data_quality_score']:.1f}%",
                f"{final_metrics['missing_data_ratio']*100:.2f}%",
                f"{final_metrics['duplicate_ratio']*100:.2f}%",
                f"{final_metrics.get('feature_quality_score', 0):.1f}%"
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
    
    def _render_transformations(self, results: Dict[str, Any]):
        """Render transformations applied"""
        st.markdown("### 🔧 Applied Transformations")
        
        preprocessing_results = results.get('detailed_results', {}).get('preprocessing_results', [])
        
        if not preprocessing_results:
            st.warning("No transformation data available.")
            return
        
        # Aggregate all transformations
        all_transformations = []
        for i, result in enumerate(preprocessing_results):
            for transformation in result['transformations_applied']:
                all_transformations.append({
                    'Iteration': i + 1,
                    'Transformation': transformation,
                    'Category': self._categorize_transformation(transformation)
                })
        
        if all_transformations:
            transform_df = pd.DataFrame(all_transformations)
            
            # Transformation category distribution
            category_counts = transform_df['Category'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(values=category_counts.values, names=category_counts.index,
                           title="Transformation Categories")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(transform_df, x='Category', title="Transformations by Category")
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed transformations table
            st.markdown("#### Detailed Transformations")
            st.dataframe(transform_df, use_container_width=True)
        else:
            st.info("No transformations were applied.")
    
    def _categorize_transformation(self, transformation: str) -> str:
        """Categorize transformation type"""
        transformation_lower = transformation.lower()
        
        if 'imputation' in transformation_lower:
            return 'Missing Data Handling'
        elif 'outlier' in transformation_lower:
            return 'Outlier Treatment'
        elif 'encoded' in transformation_lower or 'encoding' in transformation_lower:
            return 'Categorical Encoding'
        elif 'scaled' in transformation_lower or 'scaling' in transformation_lower:
            return 'Feature Scaling'
        elif 'removed' in transformation_lower or 'dropped' in transformation_lower:
            return 'Feature Removal'
        elif 'created' in transformation_lower or 'engineered' in transformation_lower:
            return 'Feature Engineering'
        elif 'converted' in transformation_lower or 'type' in transformation_lower:
            return 'Type Conversion'
        else:
            return 'Other'
    
    def _render_model_performance(self, results: Dict[str, Any]):
        """Render model performance results"""
        st.markdown("### 🤖 Model Performance")
        
        model_results = results.get('detailed_results', {}).get('training_result', {})
        
        if not model_results:
            st.warning("No model performance data available.")
            return
        
        # Best model performance
        best_model = model_results.get('best_model', {})
        if best_model.get('name'):
            st.success(f"🏆 **Best Model:** {best_model['name']}")
            
            # Performance metrics
            metrics = best_model.get('metrics', {})
            if metrics:
                col1, col2, col3 = st.columns(3)
                metric_items = list(metrics.items())
                
                for i, (metric, value) in enumerate(metric_items):
                    with [col1, col2, col3][i % 3]:
                        st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
        
        # All models comparison
        all_models = model_results.get('all_models_performance', [])
        if all_models:
            st.markdown("#### Model Comparison")
            
            model_data = []
            for model in all_models:
                model_row = {'Model': model['name']}
                model_row.update(model['metrics'])
                model_row['Training Time'] = f"{model['training_time']:.2f}s"
                model_data.append(model_row)
            
            model_df = pd.DataFrame(model_data)
            st.dataframe(model_df, use_container_width=True)
            
            # Performance visualization
            if len(all_models) > 1:
                # Get primary metric for visualization
                primary_metric = list(all_models[0]['metrics'].keys())[0]
                
                fig = px.bar(
                    x=[m['name'] for m in all_models],
                    y=[m['metrics'][primary_metric] for m in all_models],
                    title=f"Model Performance Comparison ({primary_metric.replace('_', ' ').title()})"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_recommendations(self, results: Dict[str, Any]):
        """Render recommendations and next steps"""
        st.markdown("### 💡 Recommendations & Next Steps")
        
        recommendations = results.get('recommendations', [])
        next_steps = results.get('next_steps', [])
        improvements = results.get('improvement_opportunities', {})
        
        # Recommendations
        if recommendations:
            st.markdown("#### 🎯 Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        
        # Next steps
        if next_steps:
            st.markdown("#### 📋 Next Steps")
            for step in next_steps:
                st.write(f"• {step}")
        
        # Improvement opportunities
        if improvements:
            st.markdown("#### 🚀 Improvement Opportunities")
            
            for category, opportunities in improvements.items():
                if opportunities:
                    with st.expander(f"{category.replace('_', ' ').title()} ({len(opportunities)} opportunities)"):
                        for opp in opportunities:
                            st.write(f"• {opp}")
    
    def _render_detailed_report(self, results: Dict[str, Any]):
        """Render detailed technical report"""
        st.markdown("### 📋 Detailed Technical Report")
        
        # Pipeline summary
        with st.expander("📊 Pipeline Summary"):
            summary = results.get('pipeline_summary', {})
            for key, value in summary.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        # Preprocessing details
        with st.expander("🧹 Preprocessing Details"):
            preprocessing_summary = results.get('preprocessing_summary', {})
            for key, value in preprocessing_summary.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        # Model training details
        with st.expander("🤖 Model Training Details"):
            model_summary = results.get('model_training_summary', {})
            for key, value in model_summary.items():
                if key != 'feature_importance':  # Handle separately
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        # Feature importance
        feature_importance = results.get('model_training_summary', {}).get('feature_importance', {})
        if feature_importance:
            with st.expander("🎯 Feature Importance"):
                importance_df = pd.DataFrame([
                    {'Feature': feature, 'Importance': importance}
                    for feature, importance in sorted(feature_importance.items(), 
                                                     key=lambda x: x[1], reverse=True)
                ])
                
                fig = px.bar(importance_df.head(10), x='Importance', y='Feature', 
                           orientation='h', title="Top 10 Most Important Features")
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(importance_df, use_container_width=True)
        
        # Technical configuration
        with st.expander("⚙️ Technical Configuration"):
            detailed_results = results.get('detailed_results', {})
            
            # Encoding mappings
            encodings = detailed_results.get('encoding_mappings', {})
            if encodings:
                st.write("**Encoding Mappings:**")
                st.json(encodings)
            
            # Scaling parameters
            scaling = detailed_results.get('scaling_parameters', {})
            if scaling:
                st.write("**Scaling Parameters:**")
                st.json(scaling)
            
            # Feature names
            features = detailed_results.get('final_feature_names', [])
            if features:
                st.write(f"**Final Features ({len(features)}):**")
                st.write(", ".join(features))

def render_enhanced_preprocessing_page():
    """Main function to render the enhanced preprocessing page"""
    ui = EnhancedPreprocessingUI()
    ui.render_preprocessing_page()
