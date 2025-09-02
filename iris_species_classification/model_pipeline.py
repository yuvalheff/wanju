"""
ML Pipeline for Iris Species Classification

A complete pipeline that combines data preprocessing, feature engineering,
and model prediction for deployment with MLflow.
"""

import pandas as pd
import numpy as np
from typing import Optional

from iris_species_classification.pipeline.data_preprocessing import DataProcessor
from iris_species_classification.pipeline.feature_preprocessing import FeatureProcessor
from iris_species_classification.pipeline.model import ModelWrapper


class ModelPipeline:
    """
    Complete ML pipeline for Iris Species Classification.
    
    This class combines data preprocessing, feature engineering, and model prediction
    into a single deployable pipeline suitable for MLflow model registry.
    """
    
    def __init__(self, data_processor: DataProcessor, 
                 feature_processor: FeatureProcessor, 
                 model: ModelWrapper):
        """
        Initialize the pipeline with trained components.

        Parameters:
        data_processor: Fitted DataProcessor instance
        feature_processor: Fitted FeatureProcessor instance  
        model: Fitted ModelWrapper instance
        """
        self.data_processor = data_processor
        self.feature_processor = feature_processor
        self.model = model
        
        # Validate that all components are properly fitted
        self._validate_components()
    
    def _validate_components(self):
        """Validate that all pipeline components are properly initialized."""
        if self.data_processor is None:
            raise ValueError("DataProcessor is not initialized")
        if self.feature_processor is None:
            raise ValueError("FeatureProcessor is not initialized")
        if self.model is None or self.model.model is None:
            raise ValueError("Model is not fitted")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on input data.
        
        This method handles the complete pipeline from raw input data
        to final predictions.

        Parameters:
        X: Input DataFrame with raw features

        Returns:
        np.ndarray: Predicted class labels
        """
        # Step 1: Data preprocessing (drop unnecessary columns, select features)
        X_processed = self.data_processor.transform(X)
        
        # Step 2: Feature engineering (scaling)
        X_features = self.feature_processor.transform(X_processed)
        
        # Step 3: Model prediction
        predictions = self.model.predict(X_features)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for input data.

        Parameters:
        X: Input DataFrame with raw features

        Returns:
        np.ndarray: Predicted class probabilities
        """
        # Step 1: Data preprocessing (drop unnecessary columns, select features)
        X_processed = self.data_processor.transform(X)
        
        # Step 2: Feature engineering (scaling)
        X_features = self.feature_processor.transform(X_processed)
        
        # Step 3: Model probability prediction
        probabilities = self.model.predict_proba(X_features)
        
        return probabilities
    
    def get_feature_names(self) -> list:
        """
        Get the names of features used by the model.

        Returns:
        list: List of feature names
        """
        return self.data_processor.config.feature_columns
    
    def get_model_info(self) -> dict:
        """
        Get information about the underlying model.

        Returns:
        dict: Dictionary containing model information
        """
        return {
            'model_type': self.model.config.model_type,
            'model_params': self.model.config.model_params,
            'best_params': getattr(self.model, 'best_params', None),
            'feature_columns': self.data_processor.config.feature_columns,
            'scaling_enabled': self.feature_processor.config.scaling,
            'scaler_type': self.feature_processor.config.scaler_type
        }
