import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    classification_report, f1_score, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error
)
import warnings
warnings.filterwarnings('ignore')


# --- FEATURE SETUP ---
FEATURE_COLS = ['Soil_pH', 'Temperature', 'Humidity', 'Wind_Speed', 'N', 'P', 'K', 'Soil_Quality']
SOIL_COL = 'Soil_Type'
TARGET_CLASS = 'Crop_Type'
TARGET_REG = 'Crop_Yield'

# 3 architectures to compare
ARCHITECTURES = {
    'Small  (64->32)':          (64, 32),
    'Medium (128->64->32)':     (128, 64, 32),
    'Large  (256->128->64->32)':(256, 128, 64, 32)
}


# --- PREPROCESSING ---
def preprocess(df):
    """
    Encodes Soil_Type and returns feature matrix X and both target arrays.
    """
    df_encoded = pd.get_dummies(df[FEATURE_COLS + [SOIL_COL]], columns=[SOIL_COL])
    X = df_encoded.values
    y_class = df[TARGET_CLASS].values
    y_reg = df[TARGET_REG].values
    feature_names = df_encoded.columns.tolist()
    return X, y_class, y_reg, feature_names


def split_data(X, y, stratify=None):
    """
    Splits data 70% train / 15% validation / 15% test.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42,
        stratify=stratify if stratify is not None else None
    )
    strat_temp = y_temp if stratify is not None else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42,
        stratify=strat_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# --- CLASSIFICATION MODEL ---
def train_classification_model(df):
    """
    Trains and compares 3 MLP architectures to predict Crop_Type.
    Uses 70/15/15 split + 5-fold cross validation.

    Returns dict with:
    - best_model, best_arch, best_accuracy
    - all_results: DataFrame comparing all 3 architectures
    - classification_report, confusion_matrix for best model
    - label_encoder, scaler, feature_names for prediction
    """
    X, y_class, _, feature_names = preprocess(df)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_class)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_encoded, stratify=y_encoded)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    all_results = []
    best_model = None
    best_arch_name = None
    best_val_acc = 0

    for arch_name, arch in ARCHITECTURES.items():
        model = MLPClassifier(
            hidden_layer_sizes=arch,
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        model.fit(X_train_s, y_train)

        val_acc = round(model.score(X_val_s, y_val) * 100, 2)
        test_acc = round(model.score(X_test_s, y_test) * 100, 2)
        y_pred_test = model.predict(X_test_s)
        f1 = round(f1_score(y_test, y_pred_test, average='weighted') * 100, 2)

        # 5-fold cross validation on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='accuracy')
        cv_mean = round(cv_scores.mean() * 100, 2)
        cv_std = round(cv_scores.std() * 100, 2)

        all_results.append({
            'Architecture': arch_name,
            'Val Accuracy (%)': val_acc,
            'Test Accuracy (%)': test_acc,
            'F1 Score (%)': f1,
            'CV Mean (%)': cv_mean,
            'CV Std (%)': cv_std
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_arch_name = arch_name

    y_pred_best = best_model.predict(X_test_s)
    report = classification_report(
        y_test, y_pred_best,
        target_names=le.classes_,
        output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred_best)

    return {
        "best_model": best_model,
        "best_arch": best_arch_name,
        "best_accuracy": round(best_model.score(X_test_s, y_test) * 100, 2),
        "all_results": pd.DataFrame(all_results),
        "classification_report": report,
        "confusion_matrix": cm,
        "classes": le.classes_.tolist(),
        "label_encoder": le,
        "scaler": scaler,
        "feature_names": feature_names,
        "X_test": X_test_s,
        "y_test": y_test
    }


# --- REGRESSION MODEL ---
def train_regression_model(df):
    """
    Trains and compares 3 MLP architectures to predict Crop_Yield.
    Uses 70/15/15 split + 5-fold cross validation.

    Returns dict with:
    - best_model, best_arch, best_r2, best_mae, best_rmse
    - all_results: DataFrame comparing all 3 architectures
    - y_test, y_pred for plotting actual vs predicted
    - scaler, feature_names for prediction
    """
    X, _, y_reg, feature_names = preprocess(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_reg)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    all_results = []
    best_model = None
    best_arch_name = None
    best_val_r2 = -999

    for arch_name, arch in ARCHITECTURES.items():
        model = MLPRegressor(
            hidden_layer_sizes=arch,
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        model.fit(X_train_s, y_train)

        y_pred_val = model.predict(X_val_s)
        y_pred_test = model.predict(X_test_s)

        val_r2 = round(r2_score(y_val, y_pred_val), 4)
        test_r2 = round(r2_score(y_test, y_pred_test), 4)
        mae = round(mean_absolute_error(y_test, y_pred_test), 4)
        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 4)

        # 5-fold cross validation on training set
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='r2')
        cv_mean = round(cv_scores.mean(), 4)
        cv_std = round(cv_scores.std(), 4)

        all_results.append({
            'Architecture': arch_name,
            'Val R2': val_r2,
            'Test R2': test_r2,
            'MAE': mae,
            'RMSE': rmse,
            'CV Mean R2': cv_mean,
            'CV Std R2': cv_std
        })

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model = model
            best_arch_name = arch_name

    y_pred_best = best_model.predict(X_test_s)

    return {
        "best_model": best_model,
        "best_arch": best_arch_name,
        "best_r2": round(r2_score(y_test, y_pred_best), 4),
        "best_mae": round(mean_absolute_error(y_test, y_pred_best), 4),
        "best_rmse": round(np.sqrt(mean_squared_error(y_test, y_pred_best)), 4),
        "all_results": pd.DataFrame(all_results),
        "y_test": y_test,
        "y_pred": y_pred_best,
        "scaler": scaler,
        "feature_names": feature_names,
        "X_test": X_test_s
    }


# --- SINGLE ROW PREDICTION ---
def predict_crop_type(model_results, input_data):
    """
    Predicts crop type for a single input.
    input_data: dict e.g. {"Soil_pH": 6.5, "Temperature": 25, "Soil_Type": "Loamy", ...}
    """
    model = model_results["best_model"]
    scaler = model_results["scaler"]
    le = model_results["label_encoder"]
    feature_names = model_results["feature_names"]

    row = pd.get_dummies(pd.DataFrame([input_data]), columns=[SOIL_COL])
    row = row.reindex(columns=feature_names, fill_value=0)
    row_scaled = scaler.transform(row.values)
    pred = model.predict(row_scaled)
    return le.inverse_transform(pred)[0]


def predict_crop_yield(model_results, input_data):
    """
    Predicts crop yield for a single input.
    input_data: dict e.g. {"Soil_pH": 6.5, "Temperature": 25, "Soil_Type": "Loamy", ...}
    """
    model = model_results["best_model"]
    scaler = model_results["scaler"]
    feature_names = model_results["feature_names"]

    row = pd.get_dummies(pd.DataFrame([input_data]), columns=[SOIL_COL])
    row = row.reindex(columns=feature_names, fill_value=0)
    row_scaled = scaler.transform(row.values)
    pred = model.predict(row_scaled)
    return round(float(pred[0]), 2)

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("crop_yield_dataset.csv")
    
    print("Testing Classification...")
    results = train_classification_model(df)
    print(f"Best Arch: {results['best_arch']}")
    print(f"Accuracy: {results['best_accuracy']}%")
    print(results['all_results'])
    
    print("\nTesting Regression...")
    results2 = train_regression_model(df)
    print(f"Best Arch: {results2['best_arch']}")
    print(f"R2: {results2['best_r2']}")
    print(results2['all_results'])