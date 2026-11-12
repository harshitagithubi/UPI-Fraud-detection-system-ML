
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# 1. Load dataset
data = pd.read_csv("upi_fraud_dataset.csv")

# 2. Select features
numerical_features = [
    "trans_hour",
    "trans_day",
    "trans_month",
    "age",
    "trans_amount"
]

categorical_features = [
    "category",
    "state"
]

X = data[numerical_features + categorical_features]
y = data["fraud_risk"]


# 3. Keep a separate test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# 4. Define models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Support Vector Machine": SVC()
}


# 5. Define five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}


# 6. Evaluate every model
for model_name, classifier in models.items():

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                numerical_features
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    print("\n" + "=" * 45)
    print(model_name)
    print("=" * 45)

    for metric in scoring:
        scores = results[f"test_{metric}"]

        print(
            f"{metric.capitalize()}: "
            f"{np.mean(scores):.4f} "
            f"+/- {np.std(scores):.4f}"
        )

print("\nCross-validation completed!")

