from typing import Optional
import pandas as pd
import pickle
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from iris_species_classification.config import FeaturesConfig


class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: FeaturesConfig):
        self.config: FeaturesConfig = config
        self.scaler = None
        
        if config.scaling:
            if config.scaler_type == "StandardScaler":
                self.scaler = StandardScaler()
            elif config.scaler_type == "MinMaxScaler":
                self.scaler = MinMaxScaler()
            elif config.scaler_type == "RobustScaler":
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unsupported scaler type: {config.scaler_type}")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureProcessor':
        """
        Fit the feature processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        FeatureProcessor: The fitted processor.
        """
        if self.config.scaling and self.scaler is not None:
            self.scaler.fit(X)
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input features based on the configuration.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        X_transformed = X.copy()
        
        if self.config.scaling and self.scaler is not None:
            # Apply scaling and preserve DataFrame structure
            scaled_data = self.scaler.transform(X_transformed)
            X_transformed = pd.DataFrame(scaled_data, 
                                       columns=X_transformed.columns, 
                                       index=X_transformed.index)
        
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        """
        Fit and transform the input features.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        pd.DataFrame: The transformed features.
        """
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """
        Save the feature processor as an artifact

        Parameters:
        path (str): The file path to save the processor.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'FeatureProcessor':
        """
        Load the feature processor from a saved artifact.

        Parameters:
        path (str): The file path to load the processor from.

        Returns:
        FeatureProcessor: The loaded feature processor.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
