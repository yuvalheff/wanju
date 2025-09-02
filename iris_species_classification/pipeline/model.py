import pandas as pd
import numpy as np
import pickle
import os
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

from iris_species_classification.config import ModelConfig


class ModelWrapper(BaseEstimator, ClassifierMixin):
    # Important: Declare this as a classifier for sklearn
    _estimator_type = "classifier"
    
    def __init__(self, config: ModelConfig):
        self.config: ModelConfig = config
        self.model = None
        self.best_params = None
        self._create_model()

    def _create_model(self):
        """Create the model based on configuration."""
        if self.config.model_type == "LogisticRegression":
            self.model = LogisticRegression(**self.config.model_params)
        elif self.config.model_type == "RandomForestClassifier":
            self.model = RandomForestClassifier(**self.config.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the classifier to the training data.

        Parameters:
        X: Training features.
        y: Target labels.

        Returns:
        self: Fitted classifier.
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.fit(X, y)
        # Set classes_ attribute for sklearn compatibility
        self.classes_ = self.model.classes_
        return self

    def fit_with_grid_search(self, X: pd.DataFrame, y: pd.Series, cv: int = 5):
        """
        Fit the classifier with hyperparameter tuning using GridSearchCV.

        Parameters:
        X: Training features.
        y: Target labels.
        cv: Number of cross-validation folds.

        Returns:
        self: Fitted classifier with best parameters.
        """
        if self.config.hyperparameter_grid:
            # Create base model without fitted parameters for grid search
            if self.config.model_type == "LogisticRegression":
                base_model = LogisticRegression()
                # Set fixed parameters that shouldn't be tuned
                fixed_params = {k: v for k, v in self.config.model_params.items() 
                              if k not in self.config.hyperparameter_grid}
                base_model.set_params(**fixed_params)
            elif self.config.model_type == "RandomForestClassifier":
                base_model = RandomForestClassifier()
                fixed_params = {k: v for k, v in self.config.model_params.items() 
                              if k not in self.config.hyperparameter_grid}
                base_model.set_params(**fixed_params)
            else:
                raise ValueError(f"Unsupported model type for grid search: {self.config.model_type}")
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model, 
                self.config.hyperparameter_grid, 
                cv=cv, 
                scoring='roc_auc_ovo',  # Use one-vs-one for multi-class
                n_jobs=-1
            )
            grid_search.fit(X, y)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            # Set classes_ attribute for sklearn compatibility
            self.classes_ = self.model.classes_
        else:
            # No hyperparameter tuning, just fit with default params
            self.fit(X, y)
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for the input features.

        Parameters:
        X: Input features to predict.

        Returns:
        np.ndarray: Predicted class labels.
        """
        if self.model is None:
            raise ValueError("Model not fitted")
        
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the input features.

        Parameters:
        X: Input features to predict probabilities.

        Returns:
        np.ndarray: Predicted class probabilities.
        """
        if self.model is None:
            raise ValueError("Model not fitted")
        
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importances if available.

        Returns:
        np.ndarray: Feature importances or None if not available.
        """
        if self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For logistic regression, use absolute coefficients
            return np.abs(self.model.coef_[0]) if len(self.model.coef_) == 1 else np.mean(np.abs(self.model.coef_), axis=0)
        else:
            return None

    def save(self, path: str) -> None:
        """
        Save the model wrapper as an artifact

        Parameters:
        path (str): The file path to save the model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters:
        deep: bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns:
        params: dict
            Parameter names mapped to their values.
        """
        return {'config': self.config}

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters:
        **params: dict
            Estimator parameters.

        Returns:
        self: estimator instance
            Estimator instance.
        """
        if 'config' in params:
            self.config = params['config']
            self.model = None
            self.best_params = None
            self._create_model()
        return self

    @classmethod
    def load(cls, path: str) -> 'ModelWrapper':
        """
        Load the model wrapper from a saved artifact.

        Parameters:
        path (str): The file path to load the model from.

        Returns:
        ModelWrapper: The loaded model wrapper.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)