
import sys
import os
from dotenv import load_dotenv
import requests
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
# import shap
import matplotlib.pyplot as plt

from Deliverable3.src.utils import load_artifact

load_dotenv()
st.markdown(
    """
<style>
.stApp {
background-color: #0B132B;
color: #FFFFFF;
}
[data-testid="stSidebar"] {
background-color: #0B132B;
color: #FFFFFF
}
<style>
""", unsafe_allow_html=True
)
# loading = st.empty()
# loading.image(
#     "spirograph.png",
#     width = 300
# )
# time.sleep(30)
# loading.empty()

st.title("Fake News Detection System")

# st.image("viz/custom.png", use_container_width = True)
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.write("---")
st.write("---")
st.caption(
    """This machine learning model analyzes key news to predict whether the news is a fake or real
    .Get instant results and steer your state media in the right direction."""
)
st.write("---")




FEATUREs = ['clean_text']

News_code = st.number_input("News Code", width = 100)
Station = st.text_input("Station", width = 300)
def input_features():
            clean_text = st.text_input("News Title & Text", width = 300)
            
            payload = {
                "clean_text": clean_text,
                "News_code": News_code,
                "Station": Station
            }
            features = pd.DataFrame(payload, index = [0])
            return features

df = input_features()
payloader = df.iloc[0].to_dict()
st.write(df)

API_URL = "http://127.0.0.1:8000/classify"
    
if st.button("predict"):
    try:
        response = requests.post(API_URL,params = {"Station": Station, "News_code": News_code}, json = payloader)
        response.raise_for_status()
        results = response.json()
        st.markdown("---")
        st.markdown("---")
        st.write("News Status")
        prediction = results["Prediction"]
        #st.success(f"Segment >>> {prediction}")
        report = results["Status"]
        st.markdown(report)
        st.markdown(f"Station: {Station}")
        st.markdown("---")
        st.write("Confidence Score Chart")
        # confidence = round(max(results["Probability"])* 100)

        #st.markdown(f"Confidence Score: ------------------------------------------------- {confidence}%")
     
        st.markdown("---")
        st.warning(f"Status: {results["Status"]}")
        st.markdown("---")

    except requests.exceptions.ConnectionError:
        st.error("can't reach the API server")
    except requests.exceptions.Timeout:
        st.error("Request timeout")

st.markdown("---")
#st.markdown("Silhoutte Score: ------------------------------------------------ (0.41)")
#st.write("This is a ***moderate score*** that generally reflect the ***irregular transaction behavior*** of customer in the dataset used for this system")
st.markdown("---")
#st.image("image.png", use_container_width = True)
st.markdown("<center>© 2026 JexaCore. All Rights Reserved.</center>",
           unsafe_allow_html = True)
