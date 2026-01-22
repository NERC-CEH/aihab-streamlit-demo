import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
from streamlit_js_eval import get_geolocation
import threading

# Load environment variables from .env file
load_dotenv()

# Load the API key from environment variables
api_key = os.getenv("API_KEY",None)
base_url = os.getenv("API_URL",None)
url = f"{base_url}/predict"  # Adjust if upload endpoint is different
warm_up_url = f"{base_url}/warmup"

headers = {
    "Authorization": f"Key {api_key}",
    "accept": "application/json"
}

# Warmup function this runs in a separate thread to avoid blocking the main app and to ensure the API is ready (model loaded from cache) when the user makes a request
def warmup():
    print("Warming up the API...")
    resp = requests.get(warm_up_url, headers=headers)
    print("Warmup response:", resp.status_code)
@st.cache_resource
def start_warmup_thread():
    thread = threading.Thread(target=warmup, daemon=True)
    thread.start()
    return "warmup_started"
start_warmup_thread()


st.set_page_config(page_title="AI-Hab Habitat Classifier", page_icon="static/img/ai-hab-logo-transparent.png", layout="centered")

st.title("AI-Hab Habitat Classifier")

st.markdown("AI-Hab is a habitat classification model developed by the [Laboratory of Vision Engineering](https://www.visioneng.org.uk/) at the [University of Lincoln](https://www.lincoln.ac.uk/) and the [UK Centre for Ecology & Hydrology](https://www.ceh.ac.uk/). It is based on the [UKHab](https://www.ukhab.org/) Habitat Classification system and uses computer vision to classify habitats from images. The model is trained on images from the [UKCEH Contryside Survey](https://www.ceh.ac.uk/our-science/projects/countryside-survey).") 

# Take photo or upload
st.write("## Capture or Upload Image")
img = st.camera_input("Take a photo of the habitat") or st.file_uploader("Or upload a photo", type=["png", "jpg", "jpeg"])
location = get_geolocation()

if img:
    # Send to API
    with st.spinner("Analyzing habitat..."):
        files = {"file": ("habitat.jpg", img.getvalue(), "image/jpeg")} 

        if location is not None:
            params = {"latitude": location['coords']['latitude'], 
                    "longitude": location['coords']['longitude'],
                    "top_n":3}
        else:
            params = {"top_n":3}
        resp = requests.post(url, headers=headers, params = params, files=files)
        data = resp.json()

    # Display results
    predictions = data["results"]["ukhab"]

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded image")
    with col2:
        if location is not None:
            st.map([{"lat": location['coords']['latitude'], "lon": location['coords']['longitude']}], height=150)
        else:
            st.text("Location is not enabled.")


    st.warning("**Note:** The model is still in development and may not be accurate.")

    # Show secondary predictions
    st.write("## Predictions")
    st.write("Inference time: "+ str(data["inference_time_ms"]/1000) + " seconds" )
    st.write("Message: "+ data["user_message"])
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

st.markdown("Read the preprint: [Habitat Classification from Ground-Level Imagery Using Deep Neural Networks](https://arxiv.org/abs/2507.04017).")      
st.markdown("View the code to this demonstrator app on [GitHub](https://github.com/NERC-CEH/aihab-streamlit-demo)")

# Load licence markdown (cached)
@st.cache_data
def load_licence():
    with open("static/licence/licence.md", "r", encoding="utf-8") as f:
        return f.read()

# Define a dialog using the new decorator
@st.dialog("Terms and Conditions for AI-Hab API access")
def licence_dialog():
    st.markdown(load_licence())
    st.button("Close", on_click=lambda: st.session_state.update(show_dialog=False))

# Create a button to open the dialog
if st.button("View Terms and Conditions"):
    licence_dialog()