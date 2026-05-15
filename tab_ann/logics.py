import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.inspection import permutation_importance

FEATURE_COLS = ["N", "P", "K", "Soil_pH", "Temperature", "Humidity", "Wind_Speed", "Soil_Quality"]
YIELD_COL = "Crop_Yield"
CROP_COL = "Crop_Type"


def _add_dummies(df, col):
    dummies = pd.get_dummies(df[col], prefix=col)
    return pd.concat([df, dummies], axis=1)


def preprocess_regression(df: pd.DataFrame):
    """
    Return (X, y, feature_names) for yield regression.
    Uses only environmental/soil features — Crop_Type is excluded so the model
    learns the relationship between growing conditions and yield, not crop identity.
    y is log1p-transformed to handle the heavy right-skew caused by Sugarcane.
    """
    df = df.copy().dropna(subset=FEATURE_COLS + [YIELD_COL])
    df = df[df[YIELD_COL] > 0]

    if "Soil_Type" in df.columns:
        df = _add_dummies(df, "Soil_Type")
        soil_cols = [c for c in df.columns if c.startswith("Soil_Type_")]
    else:
        soil_cols = []

    feature_names = FEATURE_COLS + soil_cols
    X = df[feature_names].values.astype(float)
    y = np.log1p(df[YIELD_COL].values.astype(float))  # log1p stabilises gradient scale
    return X, y, feature_names


def preprocess_classification(df: pd.DataFrame):
    """Return (X, y_enc, feature_names, label_encoder) for crop classification."""
    df = df.copy().dropna(subset=FEATURE_COLS + [CROP_COL])

    if "Soil_Type" in df.columns:
        df = _add_dummies(df, "Soil_Type")
        soil_cols = [c for c in df.columns if c.startswith("Soil_Type_")]
    else:
        soil_cols = []

    feature_names = FEATURE_COLS + soil_cols
    X = df[feature_names].values.astype(float)

    le = LabelEncoder()
    y = le.fit_transform(df[CROP_COL].values)
    return X, y, feature_names, le


def train_regressor(X, y, hidden_layers=(128, 64, 32), activation="relu",
                    alpha=1e-4, max_iter=500, test_size=0.2, random_state=42):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver="adam",
        alpha=alpha,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=random_state,
        verbose=False,
    )
    model.fit(X_tr_s, y_tr)

    y_pred = model.predict(X_te_s)
    mse = mean_squared_error(y_te, y_pred)

    metrics = {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_te, y_pred)),
        "r2": float(r2_score(y_te, y_pred)),
        "train_size": len(X_tr),
        "test_size": len(X_te),
        "n_iter": model.n_iter_,
        "converged": model.n_iter_ < max_iter,
    }
    return model, scaler, X_te_s, y_te, y_pred, metrics


def train_classifier(X, y, le, hidden_layers=(128, 64, 32), activation="relu",
                     alpha=1e-4, max_iter=500, test_size=0.2, random_state=42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver="adam",
        alpha=alpha,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=random_state,
        verbose=False,
    )
    model.fit(X_tr_s, y_tr)

    y_pred = model.predict(X_te_s)
    y_proba = model.predict_proba(X_te_s)

    metrics = {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "report": classification_report(y_te, y_pred, target_names=le.classes_, output_dict=True),
        "confusion_matrix": confusion_matrix(y_te, y_pred),
        "train_size": len(X_tr),
        "test_size": len(X_te),
        "n_iter": model.n_iter_,
        "converged": model.n_iter_ < max_iter
   
    }
    return model, scaler, X_te_s, y_te, y_pred, y_proba, metrics
  

def feature_importance(model, X_test, y_test, feature_names, n_repeats=8, random_state=42):
    """Permutation-based feature importance (model-agnostic, unbiased)."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    return pd.DataFrame({
        "Feature": feature_names,
        "Importance": result.importances_mean,
        "Std": result.importances_std,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)


def predict_single(model, scaler, feature_values: list):
    """Predict yield or crop class for one observation."""
    x = np.array(feature_values, dtype=float).reshape(1, -1)
    x_s = scaler.transform(x)
    pred = model.predict(x_s)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_s)[0]
    else:
        proba = None
    return pred, proba
