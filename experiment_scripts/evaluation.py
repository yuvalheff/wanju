import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.preprocessing import label_binarize

from iris_species_classification.config import ModelEvalConfig


class ModelEvaluator:
    def __init__(self, config: ModelEvalConfig):
        self.config: ModelEvalConfig = config
        
        # App color palette for consistent plotting
        self.app_color_palette = [
            'rgba(99, 110, 250, 0.8)',   # Blue
            'rgba(239, 85, 59, 0.8)',    # Red/Orange-Red
            'rgba(0, 204, 150, 0.8)',    # Green
            'rgba(171, 99, 250, 0.8)',   # Purple
            'rgba(255, 161, 90, 0.8)',   # Orange
            'rgba(25, 211, 243, 0.8)',   # Cyan
            'rgba(255, 102, 146, 0.8)',  # Pink
            'rgba(182, 232, 128, 0.8)',  # Light Green
            'rgba(255, 151, 255, 0.8)',  # Magenta
            'rgba(254, 203, 82, 0.8)'    # Yellow
        ]

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      class_names: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.

        Parameters:
        model: Trained model with predict and predict_proba methods
        X_test: Test features
        y_test: Test labels
        class_names: Names of the classes

        Returns:
        Dict[str, Any]: Dictionary containing all evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        if class_names is None:
            class_names = sorted(y_test.unique())
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        # ROC AUC for multiclass
        macro_auc = roc_auc_score(y_test, y_proba, multi_class='ovo', average='macro')
        
        # Classification report
        class_report = classification_report(y_test, y_pred, 
                                           target_names=class_names, 
                                           output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'roc_auc_macro': macro_auc,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'class_names': class_names,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        return results

    def cross_validate_model(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                           cv: int = None) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.

        Parameters:
        model: Model to evaluate (can be ModelWrapper or sklearn estimator)
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds

        Returns:
        Dict[str, Any]: Cross-validation results
        """
        if cv is None:
            cv = self.config.cv_folds
        
        # Use the underlying sklearn model if this is a ModelWrapper
        sklearn_model = model
        if hasattr(model, 'model') and model.model is not None:
            sklearn_model = model.model
        elif hasattr(model, 'model') and model.model is None:
            # Model not fitted yet, skip cross-validation
            return {
                'cv_scores': [],
                'cv_mean': 0.0,
                'cv_std': 0.0,
                'cv_folds': cv,
                'note': 'Cross-validation skipped - model not fitted'
            }
        
        # Perform cross-validation
        cv_scores = cross_val_score(sklearn_model, X_train, y_train, 
                                   cv=cv, scoring='roc_auc_ovo')
        
        results = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_folds': cv
        }
        
        return results

    def create_confusion_matrix_plot(self, cm: np.ndarray, class_names: List[str], 
                                   output_dir: str, filename: str = "confusion_matrix.html"):
        """Create confusion matrix heatmap plot."""
        fig = px.imshow(cm, 
                       x=class_names, 
                       y=class_names,
                       color_continuous_scale='Blues',
                       text_auto=True,
                       title="Confusion Matrix")
        
        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            title_x=0.5
        )
        
        self._apply_styling(fig)
        
        output_path = os.path.join(output_dir, filename)
        fig.write_html(output_path, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        
        return output_path

    def create_roc_curves_plot(self, y_test: pd.Series, y_proba: np.ndarray, 
                              class_names: List[str], output_dir: str, 
                              filename: str = "roc_curves.html"):
        """Create ROC curves for multiclass classification."""
        # Binarize the output
        y_test_bin = label_binarize(y_test, classes=class_names)
        n_classes = len(class_names)
        
        fig = go.Figure()
        
        # Plot ROC curve for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            auc_score = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{class_names[i]} (AUC = {auc_score:.3f})',
                line=dict(color=self.app_color_palette[i % len(self.app_color_palette)])
            ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='rgba(128, 128, 128, 0.8)')
        ))
        
        fig.update_layout(
            title="ROC Curves - One-vs-Rest",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            title_x=0.5
        )
        
        self._apply_styling(fig)
        
        output_path = os.path.join(output_dir, filename)
        fig.write_html(output_path, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        
        return output_path

    def create_feature_importance_plot(self, feature_names: List[str], 
                                     feature_importance: np.ndarray,
                                     output_dir: str, 
                                     filename: str = "feature_importance.html"):
        """Create feature importance plot."""
        # Sort features by importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="Feature Importance")
        
        fig.update_traces(marker_color=self.app_color_palette[0])
        fig.update_layout(title_x=0.5)
        
        self._apply_styling(fig)
        
        output_path = os.path.join(output_dir, filename)
        fig.write_html(output_path, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        
        return output_path

    def create_class_performance_plot(self, classification_report: Dict, 
                                    output_dir: str, 
                                    filename: str = "class_performance.html"):
        """Create class-wise performance metrics plot."""
        # Extract per-class metrics
        classes = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for class_name, metrics in classification_report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                classes.append(class_name)
                precision_scores.append(metrics['precision'])
                recall_scores.append(metrics['recall'])
                f1_scores.append(metrics['f1-score'])
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Precision',
            x=classes,
            y=precision_scores,
            marker_color=self.app_color_palette[0]
        ))
        
        fig.add_trace(go.Bar(
            name='Recall',
            x=classes,
            y=recall_scores,
            marker_color=self.app_color_palette[1]
        ))
        
        fig.add_trace(go.Bar(
            name='F1-Score',
            x=classes,
            y=f1_scores,
            marker_color=self.app_color_palette[2]
        ))
        
        fig.update_layout(
            title="Per-Class Performance Metrics",
            xaxis_title="Class",
            yaxis_title="Score",
            barmode='group',
            title_x=0.5
        )
        
        self._apply_styling(fig)
        
        output_path = os.path.join(output_dir, filename)
        fig.write_html(output_path, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        
        return output_path

    def create_learning_curve_plot(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                                 output_dir: str, filename: str = "learning_curve.html"):
        """Create learning curve plot."""
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, 
            cv=self.config.cv_folds,
            scoring='roc_auc_ovo',
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        # Calculate means and standard deviations
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        fig = go.Figure()
        
        # Training scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines+markers',
            name='Training Score',
            line=dict(color=self.app_color_palette[0])
        ))
        
        fig.add_trace(go.Scatter(
            x=list(train_sizes) + list(train_sizes[::-1]),
            y=list(train_mean + train_std) + list((train_mean - train_std)[::-1]),
            fill='toself',
            fillcolor='rgba(99, 110, 250, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Training ±1 std',
            showlegend=False
        ))
        
        # Validation scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_mean,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color=self.app_color_palette[1])
        ))
        
        fig.add_trace(go.Scatter(
            x=list(train_sizes) + list(train_sizes[::-1]),
            y=list(val_mean + val_std) + list((val_mean - val_std)[::-1]),
            fill='toself',
            fillcolor='rgba(239, 85, 59, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Validation ±1 std',
            showlegend=False
        ))
        
        fig.update_layout(
            title="Learning Curve",
            xaxis_title="Training Set Size",
            yaxis_title="ROC AUC Score",
            title_x=0.5
        )
        
        self._apply_styling(fig)
        
        output_path = os.path.join(output_dir, filename)
        fig.write_html(output_path, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        
        return output_path

    def _apply_styling(self, fig):
        """Apply consistent styling to plots."""
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
            font=dict(color='#8B5CF6', size=12),  # App's purple color for text
            title_font=dict(color='#7C3AED', size=16),  # Slightly darker purple for titles
            xaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',  # Purple-tinted grid
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),  # Purple tick labels
                title_font=dict(color='#7C3AED', size=12)  # Darker purple axis titles
            ),
            yaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',  # Purple-tinted grid
                zerolinecolor='rgba(139,92,246,0.3)', 
                tickfont=dict(color='#8B5CF6', size=11),  # Purple tick labels
                title_font=dict(color='#7C3AED', size=12)  # Darker purple axis titles
            ),
            legend=dict(font=dict(color='#8B5CF6', size=11))  # Purple legend
        )

    def generate_all_plots(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series,
                          feature_names: List[str], output_dir: str) -> List[str]:
        """Generate all evaluation plots and return list of created files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate model
        eval_results = self.evaluate_model(model, X_test, y_test)
        
        created_files = []
        
        # Confusion Matrix
        cm_file = self.create_confusion_matrix_plot(
            eval_results['confusion_matrix'], 
            eval_results['class_names'], 
            output_dir
        )
        created_files.append(cm_file)
        
        # ROC Curves
        roc_file = self.create_roc_curves_plot(
            y_test, eval_results['probabilities'], 
            eval_results['class_names'], output_dir
        )
        created_files.append(roc_file)
        
        # Feature Importance (if available)
        if hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance()
            if feature_importance is not None:
                fi_file = self.create_feature_importance_plot(
                    feature_names, feature_importance, output_dir
                )
                created_files.append(fi_file)
        
        # Class Performance
        cp_file = self.create_class_performance_plot(
            eval_results['classification_report'], output_dir
        )
        created_files.append(cp_file)
        
        # Learning Curve
        lc_file = self.create_learning_curve_plot(
            model, X_train, y_train, output_dir
        )
        created_files.append(lc_file)
        
        return created_files
