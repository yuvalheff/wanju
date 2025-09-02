# Experiment 1: Iris Species Classification Baseline

## Experiment Overview
**Objective**: Establish a robust baseline for multi-class iris species classification using standard machine learning approaches.

**Task Type**: Multi-class classification  
**Target Column**: Species  
**Evaluation Metric**: Macro-averaged AUC  
**Dataset**: 120 training samples, 30 test samples (perfectly balanced)

## Data Preprocessing Steps

### Feature Selection
- **Features to use**: `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`
- **Features to drop**: `Id` (provides no predictive value)
- **Rationale**: EDA and experiments confirmed all four measurement features are predictive, with petal measurements showing superior discriminative power.

### Feature Scaling
- **Method**: StandardScaler
- **Apply to**: All four measurement features
- **Rationale**: While features are on similar scales (centimeters), standardization significantly improves performance for distance-based and regularized algorithms like logistic regression.

### Missing Values
- **Strategy**: None required
- **Rationale**: Dataset contains no missing values as confirmed during EDA.

## Feature Engineering Steps

### Ratio Features (Disabled for Baseline)
- **Status**: Disabled
- **Rationale**: Initial experiments with ratio features (petal length/width, sepal length/width) achieved perfect cross-validation scores, suggesting potential overfitting. Baseline should use original features only.

### Interaction Features (Reserved)
- **Status**: Disabled
- **Rationale**: Polynomial interactions showed perfect scores but may overfit on small dataset. Reserved for future iterations.

## Model Selection Strategy

### Primary Algorithm: Logistic Regression
- **Hyperparameters**:
  - C: 1.0
  - random_state: 42
  - max_iter: 1000
  - multi_class: 'ovr'
- **Rationale**: Achieved 0.9967 macro AUC on test set. Provides interpretable coefficients and natural multi-class handling.

### Secondary Algorithm: Random Forest
- **Hyperparameters**:
  - n_estimators: 100
  - random_state: 42
  - max_depth: None
- **Rationale**: Achieved 0.9867 macro AUC. Provides feature importance insights and serves as robust tree-based comparison.

### Hyperparameter Tuning
- **Method**: GridSearchCV with 5-fold stratified cross-validation
- **Logistic Regression**: C values [0.1, 1.0, 10.0, 100.0]
- **Random Forest**: n_estimators [50, 100, 200], max_depth [None, 5, 10]
- **Scoring**: roc_auc_ovr_macro

## Evaluation Strategy

### Cross-Validation
- **Method**: 5-fold Stratified KFold
- **Configuration**: shuffle=True, random_state=42
- **Rationale**: Maintains class balance across folds with good bias-variance tradeoff for 120 samples.

### Performance Metrics
- **Primary**: Macro-averaged AUC (≥ 0.95 target)
- **Secondary**: Accuracy (≥ 0.93), per-class F1-scores (≥ 0.90)
- **Additional**: Classification report, confusion matrix, per-class AUC

### Model Interpretability Analysis

#### Feature Importance
- **Logistic Regression**: Standardized coefficients analysis
- **Random Forest**: Feature importance scores
- **Validation**: Confirm petal measurements > sepal measurements importance

#### Coefficient Analysis
- Extract and visualize standardized logistic regression coefficients
- Compare feature contributions across species classifications

### Error Analysis

#### Confusion Matrix Analysis
- Generate detailed confusion matrix with per-class metrics
- Focus on Iris-versicolor vs Iris-virginica confusion (based on EDA insights)

#### Misclassification Inspection
- Examine feature values of incorrectly classified samples
- Identify patterns in model failures

#### Calibration Analysis
- Generate reliability diagrams to assess probability calibration
- Critical for macro AUC performance evaluation

### Performance Diagnostics

#### Class-Specific Analysis
- Calculate precision, recall, F1-score per species
- Individual class AUC scores
- Identify systematically difficult species

#### Learning Curves
- Plot training/validation performance vs dataset size
- Assess potential overfitting with small dataset (120 samples)

## Expected Outputs

### Model Artifacts
- `best_logistic_regression.pkl`: Optimized logistic regression model
- `best_random_forest.pkl`: Optimized random forest model  
- `standard_scaler.pkl`: Fitted preprocessing pipeline

### Performance Reports
- `cv_results.csv`: Detailed cross-validation results
- `test_performance.json`: Final test set metrics
- `classification_report.txt`: Scikit-learn classification report
- `confusion_matrix.png`: Visualization of prediction errors

### Analysis Visualizations
- `feature_importance.png`: Comparative feature importance plot
- `logistic_coefficients.png`: Standardized coefficient visualization
- `learning_curves.png`: Training/validation learning curves
- `calibration_analysis.png`: Probability calibration assessment
- `error_analysis.md`: Detailed misclassification analysis report

## Success Criteria
- **Primary**: Macro-averaged AUC ≥ 0.95
- **Secondary**: Accuracy ≥ 0.93, all class F1-scores ≥ 0.90
- **Interpretability**: Feature importance ranking aligns with domain knowledge
- **Robustness**: Consistent CV performance (standard deviation < 0.05)

## Implementation Notes
- Use stratified sampling throughout to maintain class balance
- Set random seeds consistently (42) for reproducibility
- Focus on model interpretability given small dataset size
- Document any data quality issues or unexpected patterns
- Prepare baseline results for comparison with advanced techniques in future iterations