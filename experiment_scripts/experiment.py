import os
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import mlflow
import mlflow.sklearn
import sklearn

from iris_species_classification.pipeline.feature_preprocessing import FeatureProcessor
from iris_species_classification.pipeline.data_preprocessing import DataProcessor
from iris_species_classification.pipeline.model import ModelWrapper
from iris_species_classification.config import Config
from iris_species_classification.model_pipeline import ModelPipeline
from experiment_scripts.evaluation import ModelEvaluator

DEFAULT_CONFIG = str(Path(__file__).parent / 'config.yaml')


class Experiment:
    def __init__(self):
        self._config = Config.from_yaml(DEFAULT_CONFIG)

    def run(self, train_dataset_path, test_dataset_path, output_dir, seed=42):
        """
        Execute the complete ML experiment pipeline.

        Parameters:
        train_dataset_path: Path to training dataset
        test_dataset_path: Path to test dataset  
        output_dir: Directory to save outputs
        seed: Random seed for reproducibility

        Returns:
        dict: Experiment results in the required format
        """
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        print("🚀 Starting Iris Species Classification Experiment")
        print(f"📊 Training data: {train_dataset_path}")
        print(f"🧪 Test data: {test_dataset_path}")
        print(f"📁 Output directory: {output_dir}")
        print(f"🎲 Seed: {seed}")
        
        try:
            # Create output directories
            model_artifacts_dir = os.path.join(output_dir, "model_artifacts")
            general_artifacts_dir = os.path.join(output_dir, "general_artifacts")
            plots_dir = os.path.join(output_dir, "plots")
            
            for directory in [model_artifacts_dir, general_artifacts_dir, plots_dir]:
                os.makedirs(directory, exist_ok=True)
            
            # 1. Load data
            print("\n📖 Loading datasets...")
            train_data = pd.read_csv(train_dataset_path)
            test_data = pd.read_csv(test_dataset_path)
            
            print(f"Training data shape: {train_data.shape}")
            print(f"Test data shape: {test_data.shape}")
            
            # 2. Initialize components
            print("\n⚙️ Initializing pipeline components...")
            data_processor = DataProcessor(self._config.data_prep)
            feature_processor = FeatureProcessor(self._config.feature_prep)
            model_wrapper = ModelWrapper(self._config.model)
            evaluator = ModelEvaluator(self._config.model_evaluation)
            
            # 3. Separate features and target
            X_train = train_data.drop(columns=[self._config.data_prep.target_column])
            y_train = train_data[self._config.data_prep.target_column]
            X_test = test_data.drop(columns=[self._config.data_prep.target_column])
            y_test = test_data[self._config.data_prep.target_column]
            
            # 4. Fit preprocessing pipeline on training data
            print("\n🔄 Fitting data preprocessing pipeline...")
            data_processor.fit(X_train, y_train)
            X_train_processed = data_processor.transform(X_train)
            
            print("\n🔄 Fitting feature preprocessing pipeline...")
            feature_processor.fit(X_train_processed)
            X_train_features = feature_processor.transform(X_train_processed)
            
            # 5. Train model with hyperparameter tuning
            print("\n🎯 Training model with hyperparameter tuning...")
            model_wrapper.fit_with_grid_search(X_train_features, y_train, cv=self._config.model_evaluation.cv_folds)
            
            if model_wrapper.best_params:
                print(f"✅ Best hyperparameters: {model_wrapper.best_params}")
            
            # 6. Create complete pipeline
            print("\n🔗 Creating complete ML pipeline...")
            pipeline = ModelPipeline(
                data_processor=data_processor,
                feature_processor=feature_processor,
                model=model_wrapper
            )
            
            # 7. Test pipeline end-to-end
            print("\n🧪 Testing pipeline end-to-end...")
            # Use a small sample from training data for testing pipeline
            sample_data = X_train.head(3)
            sample_predictions = pipeline.predict(sample_data)
            sample_probabilities = pipeline.predict_proba(sample_data)
            print(f"Pipeline test successful - predictions shape: {sample_predictions.shape}")
            
            # 8. Evaluate on test set
            print("\n📊 Evaluating model on test set...")
            X_test_processed = data_processor.transform(X_test)
            X_test_features = feature_processor.transform(X_test_processed)
            
            test_evaluation = evaluator.evaluate_model(model_wrapper, X_test_features, y_test)
            
            # Primary metric (Macro AUC as specified in experiment plan)
            primary_metric_value = test_evaluation['roc_auc_macro']
            
            print(f"🎯 Test Macro AUC: {primary_metric_value:.4f}")
            print(f"🎯 Test Accuracy: {test_evaluation['accuracy']:.4f}")
            
            # 9. Cross-validation results  
            cv_results = evaluator.cross_validate_model(model_wrapper, X_train_features, y_train)
            print(f"📈 CV Macro AUC: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
            
            # 10. Generate evaluation plots
            print("\n📊 Generating evaluation plots...")
            feature_names = self._config.data_prep.feature_columns
            created_plots = evaluator.generate_all_plots(
                model_wrapper, X_train_features, y_train, 
                X_test_features, y_test, feature_names, plots_dir
            )
            print(f"📊 Generated {len(created_plots)} evaluation plots")
            
            # 11. Save individual model artifacts
            print("\n💾 Saving individual model artifacts...")
            data_processor_path = os.path.join(model_artifacts_dir, "data_processor.pkl")
            feature_processor_path = os.path.join(model_artifacts_dir, "feature_processor.pkl")
            model_path = os.path.join(model_artifacts_dir, "trained_model.pkl")
            
            data_processor.save(data_processor_path)
            feature_processor.save(feature_processor_path)
            model_wrapper.save(model_path)
            
            model_artifacts = [
                "data_processor.pkl",
                "feature_processor.pkl", 
                "trained_model.pkl"
            ]
            
            # 12. Save MLflow model
            print("\n💾 Saving MLflow model...")
            mlflow_model_dir = os.path.join(model_artifacts_dir, "mlflow_model")
            relative_path_for_return = "model_artifacts/mlflow_model/"
            
            # Create sample input for signature
            sample_input = X_train.head(2)  # Use raw input format
            sample_output = pipeline.predict(sample_input)
            
            # Always save the model to local disk for harness validation
            print(f"💾 Saving model to local disk: {mlflow_model_dir}")
            signature = mlflow.models.infer_signature(sample_input, sample_output)
            
            # Clean up existing model directory if it exists
            import shutil
            if os.path.exists(mlflow_model_dir):
                shutil.rmtree(mlflow_model_dir)
            
            mlflow.sklearn.save_model(
                pipeline,
                path=mlflow_model_dir,
                signature=signature
            )
            
            # Conditionally log to MLflow run
            active_run_id = "4da3695ac53845f98d6c844a77b00928"
            logged_model_uri = None
            
            if active_run_id and active_run_id != 'None' and active_run_id.strip():
                print(f"✅ Active MLflow run ID '{active_run_id}' detected. Logging model as artifact.")
                try:
                    with mlflow.start_run(run_id=active_run_id):
                        logged_model_info = mlflow.sklearn.log_model(
                            pipeline,
                            artifact_path="model",
                            code_paths=["iris_species_classification"],
                            signature=signature
                        )
                        logged_model_uri = logged_model_info.model_uri
                        print(f"📝 Model logged to MLflow: {logged_model_uri}")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to log to MLflow run: {e}")
            else:
                print("ℹ️ No active MLflow run ID provided. Skipping model logging.")
            
            # Add MLflow model to artifacts list
            model_artifacts.append("mlflow_model/")
            
            # 13. Save experiment results
            print("\n📄 Saving experiment results...")
            results_summary = {
                'experiment_config': {
                    'model_type': self._config.model.model_type,
                    'best_params': model_wrapper.best_params,
                    'feature_scaling': self._config.feature_prep.scaling,
                    'scaler_type': self._config.feature_prep.scaler_type
                },
                'test_results': test_evaluation,
                'cross_validation_results': cv_results,
                'feature_names': feature_names,
                'class_names': test_evaluation['class_names']
            }
            
            # Save results to general artifacts
            results_file = os.path.join(general_artifacts_dir, "experiment_results.json")
            with open(results_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                results_to_save = {}
                for key, value in results_summary.items():
                    if isinstance(value, dict):
                        results_to_save[key] = self._convert_numpy_to_lists(value)
                    else:
                        results_to_save[key] = value
                json.dump(results_to_save, f, indent=2)
            
            print("✅ Experiment completed successfully!")
            
            # 14. Return results in required format
            return {
                "metric_name": "roc_auc_macro",
                "metric_value": float(primary_metric_value),
                "model_artifacts": model_artifacts,
                "mlflow_model_info": {
                    "model_path": relative_path_for_return,
                    "logged_model_uri": logged_model_uri,
                    "model_type": "sklearn", 
                    "task_type": "classification",
                    "signature": signature.to_dict() if signature else None,
                    "framework_version": sklearn.__version__
                }
            }
            
        except Exception as e:
            print(f"❌ Experiment failed with error: {str(e)}")
            raise e

    def _convert_numpy_to_lists(self, obj):
        """Recursively convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_to_lists(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_lists(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj