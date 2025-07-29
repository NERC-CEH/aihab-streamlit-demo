import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the API key from environment variables
api_key = os.getenv("API_KEY",None)
base_url = os.getenv("API_URL",None)
url = f"{base_url}/predict"  # Adjust if upload endpoint is different

headers = {
    "Authorization": f"Key {api_key}",
    "accept": "application/json"
}

st.set_page_config(page_title="AI-Hab Habitat Classifier", page_icon="static/img/ai-hab-logo-transparent.png", layout="centered")
st.title("AI-Hab Habitat Classifier")

# Take photo or upload
img = st.camera_input("Take a photo of the habitat you wish to identify to UKHab Level 3") or st.file_uploader("Or upload a photo", type=["png", "jpg", "jpeg"])

if img:

    # Send to API
    with st.spinner("Analyzing habitat..."):
        files = {"file": ("habitat.jpg", img.getvalue(), "image/jpeg")} 
        resp = requests.post(url, headers=headers, files=files)
        data = resp.json()

    # Display results
    predictions = data["results"]["ukhab"]

    st.image(img, caption="Uploaded image")

    st.warning("**Note:** The model is still in development and may not be accurate.")

    # Show secondary predictions
    st.write("## Predictions")
    for pred in predictions[0:]:
        with st.container(border=True):
            st.subheader(f"{pred['code']} - {pred['name']}")
            if(pred['confidence']> 0.5):
                st.badge(f"**Confidence:** {pred['confidence']:.2%}",color="green")
            else:
                st.badge(f"**Confidence:** {pred['confidence']:.2%}",color="orange")
            st.write("" + " > ".join([h['name'] for h in pred['primary_habitat_hierarchy']]))

    st.write("## API Response")
    with st.expander("Show response data"):
        st.code(json.dumps(data, indent=2), language="json")



st.divider()

col1, col2 = st.columns(2)
with col1:
    st.write(" ")
    st.image("static/img/UKCEH.png")
with col2:
    st.image("static/img/University-of-Lincoln.png")

st.write("AI-Hab is a habitat classification model developed by the UK Centre for Ecology & Hydrology and the University of Lincoln. It is based on the UKHab Habitat Classification system and uses computer vision to classify habitats from images.")       
        
