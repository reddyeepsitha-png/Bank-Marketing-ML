
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/best_model.joblib")

FEATURES = [
    "age", "job", "marital", "education", "default",
    "balance", "housing", "loan", "contact", "day",
    "month", "campaign", "pdays", "previous", "poutcome"
]

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "project": "Bank Marketing Subscription Prediction",
        "model": "Random Forest selected through GridSearchCV",
        "endpoints": ["/health", "/predict"]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": "Missing fields", "fields": missing}), 400

    try:
        row = {f: data[f] for f in FEATURES}
        row["balance_per_campaign"] = (
            float(row["balance"]) / (float(row["campaign"]) + 1)
        )
        prediction = int(model.predict(pd.DataFrame([row]))[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(
                model.predict_proba(pd.DataFrame([row]))[0][1]
            )

        return jsonify({
            "prediction": "yes" if prediction == 1 else "no",
            "subscription_probability": probability
        })
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
