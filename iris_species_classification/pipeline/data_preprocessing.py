from typing import Optional
import pandas as pd
import pickle
import os

from sklearn.base import BaseEstimator, TransformerMixin

from iris_species_classification.config import DataConfig


class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: DataConfig):
        self.config: DataConfig = config

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataProcessor':
        """
        Fit the data processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        DataProcessor: The fitted processor.
        """
        # Validate that required columns exist
        missing_cols = [col for col in self.config.feature_columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data based on the configuration.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        # Make a copy to avoid modifying original data
        X_transformed = X.copy()
        
        # Drop columns specified in config
        if self.config.drop_columns:
            cols_to_drop = [col for col in self.config.drop_columns if col in X_transformed.columns]
            X_transformed = X_transformed.drop(columns=cols_to_drop)
        
        # Select only feature columns if specified
        if self.config.feature_columns:
            available_feature_cols = [col for col in self.config.feature_columns if col in X_transformed.columns]
            X_transformed = X_transformed[available_feature_cols]
        
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        """
        Fit and transform the input data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        pd.DataFrame: The transformed features.
        """
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """
        Save the data processor as an artifact

        Parameters:
        path (str): The file path to save the processor.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'DataProcessor':
        """
        Load the data processor from a saved artifact.

        Parameters:
        path (str): The file path to load the processor from.

        Returns:
        DataProcessor: The loaded data processor.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
