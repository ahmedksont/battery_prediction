from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

ALL_MODELS_PATH = "battery_all_models.joblib"

app = Flask(__name__)
CORS(app)
bundle = joblib.load(ALL_MODELS_PATH)
models = bundle["models"]
feature_cols = bundle["feature_cols"]
best_model_name = bundle["best_model_name"]
metrics = bundle["metrics"]

def build_feature_vector(
    time_h: str,
    irradiance_Wm2: float,
    solar_power_W: float,
    load_power_W: float,
) -> pd.DataFrame:
    dt = pd.to_datetime(time_h, errors="raise")
    hour = dt.hour
    dayofyear = dt.dayofyear

    data = {
        "irradiance_Wm2": [irradiance_Wm2],
        "solar_power_W": [solar_power_W],
        "load_power_W": [load_power_W],
        "hour": [hour],
        "dayofyear": [dayofyear],
    }
    X = pd.DataFrame(data)[feature_cols]
    return X


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "message": "Battery SOC prediction API",
            "target": "battery_soc_pct",
            "default_model": best_model_name,
            "available_models": list(models.keys()),
            "metrics": metrics,
            "usage": {
                "json_endpoint": {
                    "endpoint": "/predict",
                    "method": "POST",
                    "content_type": "application/json",
                    "required_fields": [
                        "time_h",
                        "irradiance_Wm2",
                        "solar_power_W",
                        "load_power_W",
                    ],
                    "optional_fields": ["model_name"],
                    "example_body": {
                        "time_h": "2025-06-10 12:00:00",
                        "irradiance_Wm2": 800.0,
                        "solar_power_W": 1500.0,
                        "load_power_W": 900.0,
                        "model_name": best_model_name,
                    },
                },
                "csv_endpoint": {
                    "endpoint": "/predict_csv",
                    "method": "POST",
                    "content_type": "multipart/form-data",
                    "form_fields": ["file", "(optional) model_name"],
                    "description": "Send a CSV file with same columns as training data; last row is used for prediction.",
                },
            },
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        required = ["time_h", "irradiance_Wm2", "solar_power_W", "load_power_W"]
        missing = [k for k in required if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        time_h = data["time_h"]
        irradiance_Wm2 = float(data["irradiance_Wm2"])
        solar_power_W = float(data["solar_power_W"])
        load_power_W = float(data["load_power_W"])

        model_name = data.get("model_name", best_model_name)
        if model_name not in models:
            return jsonify(
                {
                    "error": "model undefined",
                    "available_models": list(models.keys()),
                }
            ), 400

        model = models[model_name]

        X = build_feature_vector(
            time_h=time_h,
            irradiance_Wm2=irradiance_Wm2,
            solar_power_W=solar_power_W,
            load_power_W=load_power_W,
        )
        y_pred = model.predict(X)[0]

        return jsonify(
            {
                "model_used": model_name,
                "input": {
                    "time_h": time_h,
                    "irradiance_Wm2": irradiance_Wm2,
                    "solar_power_W": solar_power_W,
                    "load_power_W": load_power_W,
                },
                "prediction": {
                    "battery_soc_pct": float(y_pred)
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part named 'file' in request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        filename = file.filename.lower()
        if filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)
        else:
            try:
                df = pd.read_csv(file, encoding="utf-8", sep=None, engine="python")
            except UnicodeDecodeError:
                file.stream.seek(0)
                df = pd.read_csv(file, encoding="latin-1", sep=None, engine="python")
        if df is None or df.empty or df.shape[1] == 0:
            return jsonify({"error": "Uploaded file has no data or columns"}), 400

        required_cols = ["time_h", "irradiance_Wm2", "solar_power_W", "load_power_W"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            return jsonify(
                {"error": f"Missing columns in file: {missing_cols}"}
            ), 400
        last_row = df.iloc[-1]
        time_h = str(last_row["time_h"])
        irradiance_Wm2 = float(last_row["irradiance_Wm2"])
        solar_power_W = float(last_row["solar_power_W"])
        load_power_W = float(last_row["load_power_W"])
        model_name = request.form.get("model_name", best_model_name)
        if model_name not in models:
            return jsonify(
                {
                    "error": "model undefined",
                    "available_models": list(models.keys()),
                }
            ), 400

        model = models[model_name]
        X = build_feature_vector(
            time_h=time_h,
            irradiance_Wm2=irradiance_Wm2,
            solar_power_W=solar_power_W,
            load_power_W=load_power_W,
        )
        y_pred = model.predict(X)[0]

        return jsonify(
            {
                "model_used": model_name,
                "source": "file_last_row",
                "row_index": int(df.index[-1]),
                "input": {
                    "time_h": time_h,
                    "irradiance_Wm2": irradiance_Wm2,
                    "solar_power_W": solar_power_W,
                    "load_power_W": load_power_W,
                },
                "prediction": {
                    "battery_soc_pct": float(y_pred)
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
