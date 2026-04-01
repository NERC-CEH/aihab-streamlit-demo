import streamlit as st
import requests
import json
import os
import hashlib
from io import BytesIO
from datetime import datetime, timezone
from dotenv import load_dotenv
from streamlit_js_eval import get_geolocation
import threading
from huggingface_hub import HfApi
import folium
from streamlit_folium import st_folium

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

HF_BUCKET_ID = "aihab-uk/habitatimages"
HF_BUCKET_PREFIX = os.getenv("HF_BUCKET_PREFIX", "")
HABITAT_OPTIONS = [
    "u1 - Urban",
    "w1 - Broadleaved Mixed and Yew Woodland",
    "w2 - Coniferous Woodland",
    "sea - Sea",
    "c1 - Arable and Horticulture",
    "g4 - Improved Grassland",
    "g3 - Neutral Grassland",
    "g2 - Calcareous Grassland",
    "g1 - Acid Grassland",
    "g1c - Bracken",
    "h1 - Dwarf Shrub Heath",
    "f2 - Fen, Marsh, Swamp",
    "f1 - Bog",
    "t1 - Littoral Rock",
    "t2 - Littoral Sediment",
    "montane - Montane",
    "r1 - Standing Open Waters and Canals",
    "s1 - Inland Rock",
    "s2 - Supra-littoral Rock",
    "s3 - Supra-littoral Sediment",
]


def upload_bytes_to_hf_bucket(file_bytes, filename, bucket_id, bucket_prefix="", token=None):
    """Upload bytes to a Hugging Face bucket path."""
    api = HfApi(token=token)
    remote_path = f"{bucket_prefix.strip('/')}/{filename}" if bucket_prefix else filename
    api.batch_bucket_files(bucket_id, add=[(file_bytes, remote_path)])
    return remote_path


def get_hf_token():
    """Read token from process environment."""
    return os.getenv("HF_AUTH_TOKEN")


def reset_for_new_habitat():
    """Reset capture/prediction state while preserving user identity."""
    preserved_observer = st.session_state.get("observer_name_saved", "")
    new_widget_run = st.session_state.get("widget_run", 0) + 1
    st.session_state.clear()
    st.session_state["observer_name_saved"] = preserved_observer
    st.session_state["widget_run"] = new_widget_run


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

st.markdown(
    """
    <style>
    .st-key-upload_panel {
        background-color: #f8fff7;
        border: 1px solid #d9d9d9;
        border-radius: 10px;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("AI-Hab is a habitat classification model developed by the [Laboratory of Vision Engineering](https://www.visioneng.org.uk/) at the [University of Lincoln](https://www.lincoln.ac.uk/), and the [UK Centre for Ecology & Hydrology](https://www.ceh.ac.uk/). It is based on the [UKHab](https://www.ukhab.org/) Habitat Classification system and uses computer vision to classify habitats from images. The model is trained on images from the [UKCEH Contryside Survey](https://www.ceh.ac.uk/our-science/projects/countryside-survey).") 

# Take photo or upload
st.write("## Capture or Upload Image")
widget_run = st.session_state.get("widget_run", 0)
img = st.camera_input("Take a photo of the habitat", key=f"camera_{widget_run}") or st.file_uploader("Or upload a photo", type=["png", "jpg", "jpeg"], key=f"uploader_{widget_run}")
location = get_geolocation()

if img:
    image_bytes = img.getvalue()
    file_ext = os.path.splitext(getattr(img, "name", "habitat.jpg"))[1].lower()
    if file_ext not in {".jpg", ".jpeg", ".png"}:
        file_ext = ".jpg"
    image_id = f"{getattr(img, 'name', 'habitat')}-{hashlib.sha256(image_bytes).hexdigest()}"

    cached_image_id = st.session_state.get("prediction_image_id")
    if cached_image_id != image_id:
        upload_name = f"habitat_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{file_ext}"
        default_submission_datetime = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        captured_lat = location['coords']['latitude'] if location is not None else None
        captured_lon = location['coords']['longitude'] if location is not None else None

        # Send to API only for new images.
        with st.spinner("Analyzing habitat..."):
            content_type = "image/png" if file_ext == ".png" else "image/jpeg"
            files = {"file": (upload_name, image_bytes, content_type)} 

            if captured_lat is not None and captured_lon is not None:
                params = {"latitude": captured_lat,
                        "longitude": captured_lon,
                        "top_n":3}
            else:
                params = {"top_n":3}
            resp = requests.post(url, headers=headers, params = params, files=files)
            data = resp.json()

        st.session_state["prediction_image_id"] = image_id
        st.session_state["prediction_data"] = data
        st.session_state["prediction_upload_name"] = upload_name
        st.session_state["prediction_image_bytes"] = image_bytes
        st.session_state["prediction_datetime_utc"] = default_submission_datetime.isoformat()
        st.session_state["submission_date"] = default_submission_datetime.date()
        st.session_state["submission_time"] = default_submission_datetime.time().replace(second=0, microsecond=0)
        st.session_state["prediction_lat"] = captured_lat
        st.session_state["prediction_lon"] = captured_lon
        st.session_state["map_lat"] = captured_lat if captured_lat is not None else 0.0
        st.session_state["map_lon"] = captured_lon if captured_lon is not None else 0.0
        st.session_state["bucket_uploaded_for"] = None
        st.session_state["bucket_uploaded_image_path"] = None
        st.session_state["bucket_uploaded_metadata_path"] = None

    data = st.session_state.get("prediction_data")
    upload_name = st.session_state.get("prediction_upload_name")
    cached_bytes = st.session_state.get("prediction_image_bytes")
    prediction_datetime_utc = st.session_state.get("prediction_datetime_utc")
    prediction_lat = st.session_state.get("prediction_lat")
    prediction_lon = st.session_state.get("prediction_lon")

    if not data:
        st.error("Prediction failed: no response data returned.")
        st.stop()

    # Display results
    predictions = data["results"]["ukhab"]

    st.image(img, caption="Uploaded image")

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

    
    with st.container(key="upload_panel"):
        st.write("## Image Upload")
        st.info("By submitting, you confirm you have permission to upload this image and agree to the Terms and Conditions.")

        if "observer_name_saved" not in st.session_state:
            st.session_state["observer_name_saved"] = ""

        observer_name = st.text_input(
            "Name (will be saved for each new photo you add in this session)",
            value=st.session_state.get("observer_name_saved", ""),
            key="observer_name_input",
            help="Set once per session. It will be reused for all uploaded photos.",
        )
        st.session_state["observer_name_saved"] = observer_name.strip()

        top_3_predictions = [
            {
                "code": pred.get("code"),
                "name": pred.get("name"),
                "confidence": pred.get("confidence"),
            }
            for pred in predictions[:3]
        ]

        default_habitat = f"{predictions[0]['code']} - {predictions[0]['name']}"
        habitat_options = HABITAT_OPTIONS.copy()
        if default_habitat not in habitat_options:
            habitat_options.insert(0, default_habitat)

        selected_habitat = st.selectbox(
            "Select habitat label",
            options=habitat_options,
            index=habitat_options.index(default_habitat),
            help="Defaults to the top AI prediction. You can choose another habitat from the full list.",
        )
        if selected_habitat == default_habitat:
            st.caption("👍 Your current selection agrees with the AI's top prediction.")
        else:
            st.caption("Your current selection differs from the AI's top prediction.")
        selected_date = st.date_input(
            "Observation date",
            value=st.session_state.get("submission_date"),
        )
        selected_time = st.time_input(
            "Observation time (UTC)",
            value=st.session_state.get("submission_time"),
            step=60,
        )
        selected_datetime = datetime.combine(selected_date, selected_time, tzinfo=timezone.utc)
        lat_col, lon_col = st.columns(2)
        with lat_col:
            selected_lat = st.number_input(
                "Latitude",
                value=float(st.session_state.get("map_lat", prediction_lat if prediction_lat is not None else 0.0)),
                min_value=-90.0,
                max_value=90.0,
                format="%.6f",
                key="input_lat",
            )
        with lon_col:
            selected_lon = st.number_input(
                "Longitude",
                value=float(st.session_state.get("map_lon", prediction_lon if prediction_lon is not None else 0.0)),
                min_value=-180.0,
                max_value=180.0,
                format="%.6f",
                key="input_lon",
            )
        st.caption("Click the map to update the location.")
        _map = folium.Map(location=[selected_lat, selected_lon], zoom_start=13 if (selected_lat or selected_lon) else 5)
        folium.Marker([selected_lat, selected_lon]).add_to(_map)
        map_result = st_folium(_map, height=250, width="100%", returned_objects=["last_clicked"])
        if map_result and map_result.get("last_clicked"):
            clicked_lat = map_result["last_clicked"]["lat"]
            clicked_lng = map_result["last_clicked"]["lng"]
            if (clicked_lat, clicked_lng) != (st.session_state.get("map_lat"), st.session_state.get("map_lon")):
                st.session_state["map_lat"] = clicked_lat
                st.session_state["map_lon"] = clicked_lng
                st.rerun()
        comment = st.text_area("Comment", placeholder="Add an optional note about this habitat image")

        selected_habitat_parts = selected_habitat.split(" - ", 1)
        selected_habitat_code = selected_habitat_parts[0]
        selected_habitat_name = selected_habitat_parts[1] if len(selected_habitat_parts) > 1 else ""

        metadata_preview = {
            "image_file": upload_name,
            "datetime": selected_datetime.isoformat(),
            "lat": selected_lat,
            "long": selected_lon,
            "user_name": st.session_state.get("observer_name_saved", ""),
            "top_3_predictions": top_3_predictions,
            "selected_habitat_code": selected_habitat_code,
            "selected_habitat_name": selected_habitat_name,
            "comment": comment,
        }

        st.caption("Please review the Terms and Conditions before submitting.")

        already_uploaded = st.session_state.get("bucket_uploaded_for") == image_id
        if already_uploaded:
            uploaded_image_path = st.session_state.get("bucket_uploaded_image_path")
            uploaded_metadata_path = st.session_state.get("bucket_uploaded_metadata_path")
            st.success(f"Image uploaded to hf://buckets/{HF_BUCKET_ID}/{uploaded_image_path}")
            st.success(f"Metadata uploaded to hf://buckets/{HF_BUCKET_ID}/{uploaded_metadata_path}")
            if st.button("Identify a new habitat", type="secondary"):
                reset_for_new_habitat()
                st.rerun()
        elif st.button("Upload image", type="primary"):
            if not st.session_state.get("observer_name_saved", ""):
                st.error("Please enter your user name before uploading.")
                st.stop()

            hf_token = get_hf_token()
            if not hf_token:
                st.error("Missing HF token. Add HF_AUTH_TOKEN to environment variables.")
            else:
                try:
                    upload_stem = os.path.splitext(upload_name)[0]
                    remote_image_name = f"images/{upload_name}"
                    remote_metadata_name = f"metadata/{upload_stem}.json"

                    uploaded_image_path = upload_bytes_to_hf_bucket(
                        file_bytes=cached_bytes,
                        filename=remote_image_name,
                        bucket_id=HF_BUCKET_ID,
                        bucket_prefix=HF_BUCKET_PREFIX,
                        token=hf_token,
                    )

                    metadata_bytes = BytesIO(json.dumps(metadata_preview, indent=2).encode("utf-8")).getvalue()
                    uploaded_metadata_path = upload_bytes_to_hf_bucket(
                        file_bytes=metadata_bytes,
                        filename=remote_metadata_name,
                        bucket_id=HF_BUCKET_ID,
                        bucket_prefix=HF_BUCKET_PREFIX,
                        token=hf_token,
                    )

                    st.session_state["bucket_uploaded_for"] = image_id
                    st.session_state["bucket_uploaded_image_path"] = uploaded_image_path
                    st.session_state["bucket_uploaded_metadata_path"] = uploaded_metadata_path
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to upload to Hugging Face bucket: {exc}")


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

# Create a button to open the dialog
if st.button("View Terms and Conditions"):
    licence_dialog()