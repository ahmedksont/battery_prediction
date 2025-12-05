import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "solar_sousse_30days.xlsx"
BEST_MODEL_PATH = "battery_best_model.joblib"
ALL_MODELS_PATH = "battery_all_models.joblib"


def load_and_preprocess_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["time_h"] = pd.to_datetime(df["time_h"], errors="coerce")
    df["hour"] = df["time_h"].dt.hour
    df["dayofyear"] = df["time_h"].dt.dayofyear

    return df


def train_and_compare():
    df = load_and_preprocess_data(DATA_PATH)
    feature_cols = [
        "irradiance_Wm2",
        "solar_power_W",
        "load_power_W",
        "hour",
        "dayofyear",
    ]
    target_col = "battery_soc_pct"

    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "svm_rbf": make_pipeline(
            StandardScaler(),
            SVR(kernel="rbf", C=50, epsilon=0.1),
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, random_state=42
        ),
    }

    results = {}
    best_r2 = -999
    best_name = None
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"model": model, "mae": mae, "r2": r2}
        print(f"{name}: MAE = {mae:.3f} | R² = {r2:.3f}")

        if r2 > best_r2:
            best_r2 = r2
            best_name = name

    joblib.dump(
        {
            "models": {n: r["model"] for n, r in results.items()},
            "feature_cols": feature_cols,
            "metrics": {n: {"mae": r["mae"], "r2": r["r2"]} for n, r in results.items()},
            "best_model_name": best_name,
        },
        ALL_MODELS_PATH,
    )
    print(f"models saved  {ALL_MODELS_PATH}")
    joblib.dump(
        {
            "model": results[best_name]["model"],
            "feature_cols": feature_cols,
            "model_name": best_name,
        },
        BEST_MODEL_PATH,
    )
    print(f"best model saved {BEST_MODEL_PATH}")


if __name__ == "__main__":
    train_and_compare()
