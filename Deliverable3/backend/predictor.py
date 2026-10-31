
import pandas as pd
import joblib


def predict_news(news):
    df = pd.DataFrame([news])
    predictor = joblib.load("artifacts/model_mlp.pkl")
    prediction = predictor.predict(df)[0]
    probability = predictor.predict_proba(df)[0]
    if prediction == 0:
        output = "Fake News"
        alert = "Flag this as fake news"

    else:
        output = "Real News"
        alert = "This news is greenlight"

    return {
        "Prediction": output,
        "Status": alert,
        "Probability": probability.tolist()
    }

    