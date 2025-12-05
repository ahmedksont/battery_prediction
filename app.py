from flask import Flask, request, jsonify
import pandas as pd
import joblib

ALL_MODELS_PATH = "battery_all_models.joblib"

app = Flask(__name__)
bundle = joblib.load(ALL_MODELS_PATH)
models = bundle["models"]
feature_cols = bundle["feature_cols"]
bes_model_name = bundle["best_model_name"]
metrics = bundle["metrics"]


def build_feature_vector(time_h: str,
                         irradiance_Wm2: float,
                         solar_power_W: float,
                         load_power_W: float) -> pd.DataFrame:
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
            "default_model": bes_model_name,
            "available_models": list(models.keys()),
            "metrics": metrics,
            "usage": {
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
                    "model_name": bes_model_name,
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
        model_name = data.get("model_name", bes_model_name)
        if model_name not in models:
            return jsonify(
                {
                    "error": "model undifiened ",
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


if __name__ == "__main__":
    #nahi debug kn bch taamlou hebergement wele faza
    app.run(host="0.0.0.0", port=5000, debug=True)
