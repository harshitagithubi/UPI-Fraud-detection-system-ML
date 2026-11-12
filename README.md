# UPI Fraud Detection System

An end-to-end Machine Learning application that identifies potentially fraudulent UPI transactions using classification algorithms. The project combines data analysis, feature engineering, model evaluation, cross-validation and a Flask-based web application to demonstrate a complete ML workflow.

## Project Overview

Digital payment platforms have made financial transactions faster and more convenient, but they have also increased the risk of fraudulent activities.

This project explores how Machine Learning classification algorithms can be used to identify suspicious UPI transactions based on transaction-related attributes.

Four classification algorithms are implemented and evaluated:

- Random Forest Classifier
- Logistic Regression
- Decision Tree Classifier
- Support Vector Machine (SVM)

The trained Random Forest pipeline is integrated into a Flask application that allows users to enter transaction details, receive predictions and maintain transaction history using SQLite.

## Key Features

- Transaction fraud classification using Machine Learning.
- Exploratory Data Analysis and data-quality investigation.
- Feature selection based on dataset characteristics.
- Comparison of four classification algorithms.
- Evaluation using Accuracy, Precision, Recall and F1-score.
- Five-fold cross-validation for model reliability assessment.
- Random Forest feature-importance analysis.
- Interactive Flask web interface.
- SQLite-based transaction history management.
- Automatic prediction recalculation when a transaction is edited.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Data Visualization | Matplotlib, Seaborn |
| Web Framework | Flask |
| Database | SQLite |
| Model Serialization | Joblib |
| Frontend | HTML, CSS |

## Dataset Description

The dataset contains **2,666 transaction records** and 12 original columns.

The target variable is `fraud_risk`:

- `0` – Not Fraud
- `1` – Fraud

### Dataset Characteristics

| Property | Observation |
|---|---:|
| Total records | 2,666 |
| Total original columns | 12 |
| Missing values | 0 |
| Duplicate complete rows | 0 |
| Fraud transactions (1) | 1,578 |
| Non-fraud transactions (0) | 1,088 |
| Fraud class percentage | 59.19% |
| Non-fraud class percentage | 40.81% |

The dataset was examined for missing values, duplicate records, unique values, target distribution and relationships between individual features and the target variable.

## Exploratory Data Analysis and Feature Analysis

Before training the models, the dataset was analysed to understand its structure and identify potentially unsuitable features.

### 1. Data Quality Analysis

The following checks were performed:

- Missing-value analysis.
- Duplicate-row detection.
- Unique-value analysis for individual columns.
- Duplicate transaction ID investigation.
- Target distribution analysis.
- Category-wise and state-wise fraud-rate analysis.
- Transaction amount comparison across target classes.

An important observation was that two transaction ID values occurred more than once, and one repeated ID was associated with conflicting target labels.

### 2. Transaction Amount Analysis

The mean transaction amounts across the two target classes were:

| Transaction Class | Mean Transaction Amount |
|---|---:|
| Not Fraud (0) | 66.85 |
| Fraud (1) | 511.17 |

This indicated a substantial difference in transaction amounts between the two classes in this dataset.

### 3. Feature Selection

The original dataset contained identifiers and constant or high-cardinality columns that required investigation before model training.

The following columns were excluded:

| Feature | Reason for Exclusion |
|---|---|
| `Id` | Identifier rather than a meaningful transaction characteristic; also contained repeated values |
| `trans_year` | Constant value across the dataset |
| `upi_number` | Unique identifier for every record; unsuitable for generalising to unseen transactions |
| `zip` | High-cardinality feature with 634 unique values and 223 values appearing only once |

### Final Features Used

Seven features were selected for model training.

| Feature | Type | Description |
|---|---|---|
| `trans_hour` | Numerical | Hour of transaction |
| `trans_day` | Numerical | Day of transaction |
| `trans_month` | Numerical | Month of transaction |
| `trans_amount` | Numerical | Transaction amount |
| `age` | Numerical | User age |
| `category` | Categorical | Transaction category code |
| `state` | Categorical | State code |

## Data Preprocessing

A Scikit-learn preprocessing pipeline was implemented to ensure consistent transformations during training and prediction.

### Numerical Features

`StandardScaler` was used to standardise numerical features.

Standardisation transforms numerical values into a common scale, which is particularly useful for algorithms such as Logistic Regression and Support Vector Machines.

### Categorical Features

`OneHotEncoder` was used to convert categorical features into numerical representations.

The encoder was configured with `handle_unknown="ignore"` so that previously unseen categories would not cause prediction-time errors.

### Pipeline Implementation

The preprocessing steps were integrated with the classifiers using Scikit-learn's `Pipeline` and `ColumnTransformer`.

This ensures that preprocessing is fitted on training data and applied consistently to testing data and future transactions.

## Model Training

The dataset was divided into training and testing subsets.

| Parameter | Value |
|---|---:|
| Training set | 1,866 records |
| Testing set | 800 records |
| Training percentage | 70% |
| Testing percentage | 30% |
| Random state | 42 |
| Stratification | Enabled |

Four classification algorithms were trained and evaluated using the same selected features and preprocessing approach.

### Algorithms Implemented

**1. Random Forest Classifier**

An ensemble learning algorithm consisting of multiple decision trees. It combines individual tree predictions through voting to produce the final classification.

Configuration: 100 estimators, random state 42.

**2. Logistic Regression**

A supervised classification algorithm that estimates class probabilities using a logistic function. It provides a useful baseline for comparison.

**3. Decision Tree Classifier**

A tree-based algorithm that learns decision rules by recursively splitting the dataset according to feature values.

**4. Support Vector Machine**

A supervised learning algorithm that identifies a decision boundary between classes. The default RBF kernel was used.

## Model Evaluation Results

The models were evaluated using the following metrics:

- **Accuracy:** Overall proportion of correctly classified transactions.
- **Precision:** Proportion of predicted fraud transactions that were actually labelled fraud.
- **Recall:** Proportion of actual fraud transactions correctly identified.
- **F1-score:** Harmonic mean of precision and recall.

### Test Set Performance

The following results were obtained from the 800-record test set.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | **97.00%** | **97.87%** | **97.05%** | **97.46%** |
| Logistic Regression | 78.63% | 83.44% | 79.75% | 81.55% |
| Decision Tree | 94.25% | 95.15% | 95.15% | 95.15% |
| Support Vector Machine | 88.12% | 90.41% | 89.45% | 89.93% |

**Observation:** Random Forest achieved the highest performance across all four reported metrics on the test set.

The Decision Tree also performed strongly, while Logistic Regression provided a comparatively weaker baseline.

### Confusion Matrix Analysis

The Random Forest confusion matrix was:

| | Predicted Not Fraud | Predicted Fraud |
|---|---:|---:|
| Actual Not Fraud | 316 | 10 |
| Actual Fraud | 14 | 460 |

Interpretation:

- True Negatives: 316
- False Positives: 10
- False Negatives: 14
- True Positives: 460

The model correctly classified 776 out of 800 test transactions.

## Cross-Validation

To evaluate model consistency beyond a single train-test split, five-fold cross-validation was performed.

The dataset was divided into five folds. In each iteration, four folds were used for training and the remaining fold was used for validation.

Each fold served as the validation set once.

The mean and standard deviation of the evaluation metrics were calculated across the five folds.

### Five-Fold Cross-Validation Results

| Model | Accuracy (Mean ± SD) | Precision (Mean ± SD) | Recall (Mean ± SD) | F1-score (Mean ± SD) |
|---|---:|---:|---:|---:|
| Random Forest | **95.66 ± 1.01%** | **96.63 ± 0.35%** | **96.02 ± 1.70%** | **96.31 ± 0.89%** |
| Logistic Regression | 80.33 ± 2.19% | 84.99 ± 0.73% | 81.07 ± 4.11% | 82.93 ± 2.30% |
| Decision Tree | 94.27 ± 1.72% | 94.48 ± 1.59% | 95.93 ± 2.00% | 95.19 ± 1.46% |
| Support Vector Machine | 88.32 ± 1.01% | 90.84 ± 1.43% | 89.31 ± 2.43% | 90.03 ± 0.94% |

*Values are expressed as percentage mean ± percentage-point standard deviation.*

### Cross-Validation Observations

- Random Forest achieved the highest mean accuracy and F1-score.
- Its relatively small standard deviations indicate consistent performance across the five folds.
- Decision Tree also showed strong performance.
- Logistic Regression showed comparatively greater variation in recall.
- Cross-validation provides a more robust assessment than relying on one train-test split alone.

### Test Set vs Cross-Validation Accuracy

| Model | Test Accuracy | CV Mean Accuracy |
|---|---:|---:|
| Random Forest | 97.00% | 95.66% |
| Logistic Regression | 78.63% | 80.33% |
| Decision Tree | 94.25% | 94.27% |
| Support Vector Machine | 88.12% | 88.32% |

The difference between test accuracy and cross-validation accuracy reflects the use of different evaluation samples and procedures.

## Feature Importance Analysis

Feature importance was extracted from the trained Random Forest pipeline to understand which input variables contributed most to the model's predictions.

The analysis was performed using the Random Forest classifier's `feature_importances_` attribute after preprocessing.

The top features identified were:

1. Transaction Amount
2. Transaction Hour
3. Age
4. Transaction Day
5. Transaction Month
6. Transaction Category
7. State

### Key Observations

- Transaction amount was the most influential feature.
- Transaction hour was the second most influential feature.
- Age and transaction day had smaller contributions.
- Individual one-hot encoded category features also contributed to the model.

Feature importance represents the relative contribution of features to the tree-based model's splitting decisions. It does not establish causation or guarantee that a feature independently predicts fraud.

## Flask Web Application

The trained Random Forest pipeline was integrated into a Flask application to provide an interactive interface for transaction classification.

### Application Workflow

1. Users enter transaction details through the web form.
2. Flask receives and validates the submitted values.
3. The input is converted into a Pandas DataFrame.
4. The saved preprocessing and Random Forest pipeline processes the input.
5. The model predicts the transaction class.
6. The result is displayed as Fraud or Not Fraud.
7. Submitted transactions are stored in an SQLite database.

### Application Features

- Transaction prediction.
- Transaction submission.
- Transaction history.
- Editing existing transactions.
- Recalculation of predictions after editing.
- Deletion of transaction records.

The trained model is saved using Joblib and loaded by the Flask application for inference.

## Project Structure

```text
UPI-FRAUD-DETECTION/
│
├── app.py
├── 4algos.py
├── 4algos_improved.py
├── cross_validation.py
├── feature_importance.py
├── data_quality_check.py
├── feature_analysis.py
├── explore_data.py
│
├── upi_fraud_dataset.csv
│
├── random_forest_pipeline.pkl
├── decision_tree_pipeline.pkl
├── logistic_regression_pipeline.pkl
├── support_vector_machine_pipeline.pkl
│
├── transactions.db
│
├── templates/
│   ├── index.html
│   ├── history.html
│   └── edit.html
│
├── static/
│   └── styles.css
│
├── assets/
│   ├── model_comparison.png
│   ├── random_forest_confusion_matrix.png
│   ├── logistic_regression_confusion_matrix.png
│   ├── decision_tree_confusion_matrix.png
│   ├── svm_confusion_matrix.png
│   └── feature_importance.png
│
├── README.md
└── .gitignore
```

The assets folder contains visual outputs from model evaluation and feature analysis.

## Installation and Execution

### 1. Clone the repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd UPI-FRAUD-DETECTION
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

For macOS/Linux:

```bash
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install flask pandas numpy scikit-learn matplotlib seaborn joblib
```

For compatibility with the saved model files, use the same scikit-learn version used when they were generated. The model files should be regenerated if the environment's scikit-learn version changes incompatibly.

### 5. Run the Flask application

```bash
python3 app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5000/
```

### Retraining the Models

To train and evaluate the four classifiers:

```bash
python3 4algos_improved.py
```

### Running Cross-Validation

```bash
python3 cross_validation.py
```

### Running Feature Importance Analysis

```bash
python3 feature_importance.py
```

## Improvements and Engineering Decisions

Several improvements were implemented to make the ML workflow more reliable and the application more consistent.

| Area | Improvement |
|---|---|
| Data exploration | Added structured dataset and data-quality analysis |
| Feature selection | Investigated and excluded unsuitable identifier and constant columns |
| Preprocessing | Used StandardScaler and OneHotEncoder |
| Data leakage prevention | Fitted preprocessing within pipelines after train-test splitting |
| Model evaluation | Added precision, recall, F1-score and classification reports |
| Model validation | Added five-fold cross-validation |
| Feature analysis | Added Random Forest feature-importance visualisation |
| Model persistence | Saved complete preprocessing and classifier pipelines |
| Flask integration | Connected the improved Random Forest pipeline |
| Database | Updated transaction storage to support all seven model features |
| Editing functionality | Recalculates the prediction after transaction updates |

## Limitations

- Model performance depends on the quality and representativeness of the available dataset.
- The dataset contains only 2,666 records, so results may not generalise to larger real-world transaction streams.
- High feature importance does not establish causality.
- The system does not connect to banking networks or verify actual UPI payments.
- Predictions are model classifications, not confirmed fraud investigations.
- Further testing on independently sourced, representative transaction data would be necessary before real-world deployment.

## Future Improvements

- Evaluate the model on larger and independently sourced transaction datasets.
- Introduce probability-based risk scores and configurable decision thresholds.
- Explore hyperparameter tuning.
- Improve explainability with SHAP or other model-interpretation techniques.
- Add stronger application security and user authentication.
- Consider deployment using a production-grade WSGI server.

## Conclusion

This project demonstrates an end-to-end Machine Learning workflow for UPI transaction fraud classification, covering dataset exploration, feature selection, preprocessing, comparative model training, evaluation, cross-validation, feature importance and web application integration.

Among the four evaluated algorithms, Random Forest achieved the strongest test-set performance, with 97.00% accuracy and a 97.46% F1-score. Its five-fold cross-validation accuracy was 95.66% ± 1.01%.

The project highlights the importance of evaluating models using multiple metrics, validating performance across different data splits and integrating trained ML pipelines into practical applications.



AUTHOR
HARSHITA GUPTA 
BTECH CSE AI IGDTUW
Harshitaigcs@gmail.com
