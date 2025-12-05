Install dependencies
pip install flask pandas scikit-learn joblib openpyxl

Training the Models

Run the training script:

python train_models.py

Running the Flask API

Start the server:

python app.py


API will run at:

http://127.0.0.1:5000/
POST /predict

Send JSON input:

{
  "time_h": "2025-06-10 12:00:00",
  "irradiance_Wm2": 800,
  "solar_power_W": 2500,
  "load_power_W": 1000
}

Example Response
{
  "model_used": "random_forest",
  "prediction": {
    "battery_soc_pct": 100.0
  }
}

📄 License

This project is open-source and free to use for any purpose.

⭐ Support the Project

If you found this useful, please give the repository a star ⭐ on GitHub!
