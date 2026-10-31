
"""This module contain function for testing the saved model"""


import pandas as pd
from Deliverable3.src.utils import load_artifact


feature = ["clean_text"]


def predict(saved_model_path: str, record: dict)->dict:

    model = load_artifact(saved_model_path)
    X = pd.DataFrame([record], columns = feature)
    prediction = model.predict(X)[0]
    return {
        "prediction": int(prediction),
        "label": "Fake News" if prediction == 0 else "Real News"}



if __name__ == "__main__":
    news = {'clean_text': """Trump is now vowing to win enough delegates to clinch the Republican nomination and prevent a contested convention. But Sens.Ted Cruz, R-Texas, Bernie Sanders, D-Vt., and Ohio Gov. John Kasich and aren't giving up just yet.

A big win in New York could tip the scales for both the Republican and Democratic front-runners in this year's race for the White House. Clinton and Trump have each suffered losses in recent contests, shifting the momentum to their rivals.

We have won eight out of the last nine caucuses and primaries! Cheer! Sanders recently told supporters.

While wins in New York for Trump and Clinton are expected, the margins of those victories are also important.

Trump needs to capture more than 50 percent of the vote statewide if he wants to be positioned to win all of the state's 95 GOP delegates. That would put him one step closer to avoiding a contested convention."""}

    result = predict("artifacts/model_mlp.pkl", news)
    print(result)