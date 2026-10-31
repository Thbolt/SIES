import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ==================================================
# SETTINGS
# ==================================================

DATA_PATH = "data/paysim.csv"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)


# ==================================================
# LOAD DATA
# ==================================================

print("\nLoading PaySim dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==================================================
# REMOVE UNNECESSARY COLUMNS
# ==================================================

columns_to_drop = [
    "nameOrig",
    "nameDest",
    "isFlaggedFraud"
]

df = df.drop(
    columns=[
        c for c in columns_to_drop
        if c in df.columns
    ],
    errors="ignore"
)


# ==================================================
# FEATURE ENGINEERING
# ==================================================

print("\nCreating features...")


# Difference between sender's old and new balance
if (
    "oldbalanceOrg" in df.columns
    and "newbalanceOrig" in df.columns
):

    df["balance_change"] = (
        df["oldbalanceOrg"]
        -
        df["newbalanceOrig"]
    )


# Difference between transaction amount
# and sender balance
if (
    "amount" in df.columns
    and "oldbalanceOrg" in df.columns
):

    df["amount_to_balance_ratio"] = (
        df["amount"]
        /
        (df["oldbalanceOrg"] + 1)
    )


# Destination balance change
if (
    "oldbalanceDest" in df.columns
    and "newbalanceDest" in df.columns
):

    df["destination_balance_change"] = (
        df["newbalanceDest"]
        -
        df["oldbalanceDest"]
    )


# ==================================================
# TARGET
# ==================================================

TARGET = "isFraud"

if TARGET not in df.columns:

    raise ValueError(
        "The dataset does not contain the 'isFraud' column."
    )


X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]


# ==================================================
# DISPLAY CLASS DISTRIBUTION
# ==================================================

print("\nFraud distribution:")

print(
    y.value_counts()
)

print("\nFraud percentage:")

print(
    y.value_counts(normalize=True) * 100
)


# ==================================================
# IDENTIFY FEATURE TYPES
# ==================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# ==================================================
# PREPROCESSING
# ==================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numerical_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==================================================
# MODEL 1 - LOGISTIC REGRESSION
# ==================================================

print("\n" + "=" * 50)
print("Training Logistic Regression...")
print("=" * 50)


logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)


logistic_model.fit(
    X_train,
    y_train
)


logistic_predictions = logistic_model.predict(
    X_test
)

logistic_probabilities = logistic_model.predict_proba(
    X_test
)[:, 1]


print("\nLOGISTIC REGRESSION")

print(
    classification_report(
        y_test,
        logistic_predictions,
        zero_division=0
    )
)

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        logistic_probabilities
    )
)

print(
    "PR-AUC:",
    average_precision_score(
        y_test,
        logistic_probabilities
    )
)


# ==================================================
# MODEL 2 - RANDOM FOREST
# ==================================================

print("\n" + "=" * 50)
print("Training Random Forest...")
print("=" * 50)


random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=15,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


random_forest_model.fit(
    X_train,
    y_train
)


rf_predictions = random_forest_model.predict(
    X_test
)

rf_probabilities = random_forest_model.predict_proba(
    X_test
)[:, 1]


print("\nRANDOM FOREST")

print(
    classification_report(
        y_test,
        rf_predictions,
        zero_division=0
    )
)

rf_roc_auc = roc_auc_score(
    y_test,
    rf_probabilities
)

rf_pr_auc = average_precision_score(
    y_test,
    rf_probabilities
)

print(
    "ROC-AUC:",
    rf_roc_auc
)

print(
    "PR-AUC:",
    rf_pr_auc
)


# ==================================================
# CONFUSION MATRIX
# ==================================================

print("\nRandom Forest Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        rf_predictions
    )
)


# ==================================================
# SAVE BEST MODEL
# ==================================================

# For fraud detection we use PR-AUC as an important
# comparison metric because the fraud class is highly
# imbalanced.

logistic_pr_auc = average_precision_score(
    y_test,
    logistic_probabilities
)


if rf_pr_auc >= logistic_pr_auc:

    best_model = random_forest_model

    print(
        "\nRandom Forest selected as the final model."
    )

else:

    best_model = logistic_model

    print(
        "\nLogistic Regression selected as the final model."
    )


model_path = os.path.join(
    MODEL_DIR,
    "fraud_model.pkl"
)


joblib.dump(
    best_model,
    model_path
)


print("\nModel saved successfully:")
print(model_path)

print("\nTraining completed.")