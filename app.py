import streamlit as st
import requests
import json
import os
import hashlib
from io import BytesIO
from datetime import datetime, timezone
from dotenv import load_dotenv
from streamlit_js_eval import get_geolocation, streamlit_js_eval
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

CARDINAL_DIRECTION_TO_DEGREES = {
    "N": 0.0,
    "NE": 45.0,
    "E": 90.0,
    "SE": 135.0,
    "S": 180.0,
    "SW": 225.0,
    "W": 270.0,
    "NW": 315.0,
}
CARDINAL_DIRECTIONS = list(CARDINAL_DIRECTION_TO_DEGREES.keys())


## Convert numeric heading to nearest cardinal/intercardinal direction.
def heading_to_cardinal(heading):
    """Map heading in degrees to nearest cardinal/intercardinal direction."""
    if heading is None:
        return None
    normalized = ((float(heading) % 360) + 360) % 360
    idx = int((normalized + 22.5) // 45) % len(CARDINAL_DIRECTIONS)
    return CARDINAL_DIRECTIONS[idx]


## Read compass heading from browser device-orientation APIs.
def get_compass_heading(measurement_key):
    """Attempt to read device compass heading in degrees (0-360, where 0 is north)."""
    js_expression = """
        (async () => {
            const normalize = (value) => {
                if (value === null || value === undefined || Number.isNaN(value)) {
                    return null;
                }
                const n = ((Number(value) % 360) + 360) % 360;
                return Math.round(n * 10) / 10;
            };

            const getScreenOrientationAngle = () => {
                try {
                    if (typeof screen !== "undefined" && screen.orientation && typeof screen.orientation.angle === "number") {
                        return screen.orientation.angle;
                    }
                    if (typeof window !== "undefined" && typeof window.orientation === "number") {
                        return window.orientation;
                    }
                } catch (_err) {
                    return 0;
                }
                return 0;
            };

            try {
                if (typeof window === "undefined" || typeof DeviceOrientationEvent === "undefined") {
                    return null;
                }

                const getSingleHeading = () => new Promise((resolve) => {
                    let settled = false;

                    const finish = (heading) => {
                        if (settled) {
                            return;
                        }
                        settled = true;
                        resolve(normalize(heading));
                    };

                    const handler = (event) => {
                        const webkitHeading = event.webkitCompassHeading;
                        const alphaHeading =
                            event.alpha !== null && event.alpha !== undefined
                                ? (360 - Number(event.alpha)) + getScreenOrientationAngle()
                                : null;
                        const heading =
                            webkitHeading !== undefined && webkitHeading !== null
                                ? webkitHeading
                                : alphaHeading;
                        finish(heading);
                    };

                    window.addEventListener("deviceorientationabsolute", handler, { once: true });
                    window.addEventListener("deviceorientation", handler, { once: true });
                    setTimeout(() => finish(null), 2000);
                });

                if (typeof DeviceOrientationEvent.requestPermission === "function") {
                    const permission = await DeviceOrientationEvent.requestPermission();
                    if (permission !== "granted") {
                        return null;
                    }
                }

                return await getSingleHeading();
            } catch (_err) {
                return null;
            }
        })()
    """

    heading = streamlit_js_eval(
        js_expressions=js_expression,
        key=measurement_key,
    )

    if isinstance(heading, (int, float)):
        return float(heading)
    return None


## Upload raw bytes to the configured Hugging Face bucket path.
def upload_bytes_to_hf_bucket(file_bytes, filename, bucket_id, bucket_prefix="", token=None):
    """Upload bytes to a Hugging Face bucket path."""
    api = HfApi(token=token)
    remote_path = f"{bucket_prefix.strip('/')}/{filename}" if bucket_prefix else filename
    api.batch_bucket_files(bucket_id, add=[(file_bytes, remote_path)])
    return remote_path


## Fetch Hugging Face auth token from environment.
def get_hf_token():
    """Read token from process environment."""
    return os.getenv("HF_AUTH_TOKEN")


## Clear per-capture state while keeping observer name between captures.
def reset_for_new_habitat():
    """Reset capture/prediction state while preserving user identity."""
    preserved_observer = st.session_state.get("observer_name_saved", "")
    new_widget_run = st.session_state.get("widget_run", 0) + 1
    st.session_state.clear()
    st.session_state["observer_name_saved"] = preserved_observer
    st.session_state["widget_run"] = new_widget_run


## Normalize image extension to supported upload formats.
def get_image_file_ext(image_file):
    """Get a safe image extension for API upload content-type handling."""
    file_ext = os.path.splitext(getattr(image_file, "name", "habitat.jpg"))[1].lower()
    return file_ext if file_ext in {".jpg", ".jpeg", ".png"} else ".jpg"


## Build cache key from filename and file content hash.
def build_image_id(image_file, image_bytes):
    """Create a deterministic image id for session caching."""
    return f"{getattr(image_file, 'name', 'habitat')}-{hashlib.sha256(image_bytes).hexdigest()}"


## Safely extract latitude/longitude from geolocation payload.
def get_location_coords(location):
    """Extract lat/lon if available from browser geolocation payload."""
    if not location or "coords" not in location:
        return None, None
    return location["coords"].get("latitude"), location["coords"].get("longitude")


## Submit image and optional coordinates to prediction API.
def request_prediction(upload_name, image_bytes, file_ext, captured_lat, captured_lon):
    """Call prediction API and return parsed JSON response."""
    content_type = "image/png" if file_ext == ".png" else "image/jpeg"
    files = {"file": (upload_name, image_bytes, content_type)}
    params = {"top_n": 3}
    if captured_lat is not None and captured_lon is not None:
        params.update({"latitude": captured_lat, "longitude": captured_lon})

    resp = requests.post(url, headers=headers, params=params, files=files)
    return resp.json()


## Persist prediction payload and default submission values in session state.
def cache_prediction_state(
    image_id,
    data,
    upload_name,
    image_bytes,
    default_submission_datetime,
    captured_lat,
    captured_lon,
    captured_heading,
):
    """Persist prediction and submission defaults in session state."""
    st.session_state["prediction_image_id"] = image_id
    st.session_state["prediction_data"] = data
    st.session_state["prediction_upload_name"] = upload_name
    st.session_state["prediction_image_bytes"] = image_bytes
    st.session_state["prediction_datetime_utc"] = default_submission_datetime.isoformat()
    st.session_state["submission_date"] = default_submission_datetime.date()
    st.session_state["submission_time"] = default_submission_datetime.time().replace(second=0, microsecond=0)
    st.session_state["prediction_lat"] = captured_lat
    st.session_state["prediction_lon"] = captured_lon
    st.session_state["prediction_heading"] = captured_heading
    st.session_state["map_lat"] = captured_lat if captured_lat is not None else 0.0
    st.session_state["map_lon"] = captured_lon if captured_lon is not None else 0.0
    st.session_state["bucket_uploaded_for"] = None
    st.session_state["bucket_uploaded_image_path"] = None
    st.session_state["bucket_uploaded_metadata_path"] = None


## Pick badge color from confidence threshold.
def confidence_badge_color(confidence):
    """Color confidence badge using a fixed threshold."""
    return "green" if confidence > 0.5 else "orange"


## Build compact top-N prediction list for metadata.
def extract_top_predictions(predictions, limit=3):
    """Keep a compact top-N summary for metadata upload."""
    return [
        {
            "code": pred.get("code"),
            "name": pred.get("name"),
            "confidence": pred.get("confidence"),
        }
        for pred in predictions[:limit]
    ]


## Render the predictions card panel and inference timing.
def render_predictions_panel(predictions, inference_time_ms):
    """Render prediction cards in a dedicated styled panel."""
    with st.container(key="predictions_panel"):
        st.write("## Predictions")
        for pred in predictions:
            title_col, badge_col = st.columns([4, 2])
            with title_col:
                st.subheader(f"{pred['code']} - {pred['name']}")
            with badge_col:
                st.badge(
                    f"**Confidence:** {pred['confidence']:.2%}",
                    color=confidence_badge_color(pred["confidence"]),
                )
            st.write("" + " > ".join([h["name"] for h in pred["primary_habitat_hierarchy"]]))
        st.caption(f"Inference time: {inference_time_ms/1000} seconds")


## Construct metadata JSON object for bucket upload.
def build_metadata_preview(
    upload_name,
    selected_datetime,
    selected_lat,
    selected_lon,
    selected_heading,
    selected_direction_value,
    bearing_source,
    observer_name,
    top_3_predictions,
    selected_habitat_code,
    selected_habitat_name,
    comment,
):
    """Build metadata document uploaded with each submitted image."""
    return {
        "image_file": upload_name,
        "datetime": selected_datetime.isoformat(),
        "lat": selected_lat,
        "long": selected_lon,
        "bearing_degrees": selected_heading,
        "bearing_cardinal": selected_direction_value,
        "bearing_source": bearing_source,
        "user_name": observer_name,
        "top_3_predictions": top_3_predictions,
        "selected_habitat_code": selected_habitat_code,
        "selected_habitat_name": selected_habitat_name,
        "comment": comment,
    }


## Render upload form and handle final image/metadata submission.
def render_submission_panel(
    image_id,
    predictions,
    upload_name,
    cached_bytes,
    prediction_lat,
    prediction_lon,
    prediction_heading,
):
    """Render user submission controls and perform bucket upload on submit."""
    with st.container(key="upload_panel"):
        st.write("## Submit observation")
        st.info("Your image will be used to evaluate and improve the AI-Hab model. By submitting, you confirm you have permission to upload this image and agree to the Terms and Conditions.")

        if "observer_name_saved" not in st.session_state:
            st.session_state["observer_name_saved"] = ""

        observer_name = st.text_input(
            "Name (will be saved for each new photo you add in this session)",
            value=st.session_state.get("observer_name_saved", ""),
            key="observer_name_input",
            help="Set once per session. It will be reused for all uploaded photos.",
        )
        st.session_state["observer_name_saved"] = observer_name.strip()

        top_3_predictions = extract_top_predictions(predictions)

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

        # Temporarily disable compass/direction UI and metadata until heading is reliable.
        # auto_direction = heading_to_cardinal(prediction_heading)
        # direction_options = ["", *CARDINAL_DIRECTIONS]
        # default_direction = auto_direction if auto_direction else ""
        # selected_direction = st.selectbox(
        #     "Photo direction",
        #     options=direction_options,
        #     index=direction_options.index(default_direction),
        #     help="Direction classes only (N/NE/E/SE/S/SW/W/NW). Auto-filled from compass when available.",
        # )
        # selected_direction_value = selected_direction or None
        # selected_heading = CARDINAL_DIRECTION_TO_DEGREES.get(selected_direction_value)
        #
        # if auto_direction:
        #     st.caption(f"Auto-detected direction from compass: {auto_direction}")
        #     if selected_direction_value and selected_direction_value != auto_direction:
        #         bearing_source = "manual_override_cardinal"
        #     elif selected_direction_value:
        #         bearing_source = "sensor_translated_cardinal"
        #     else:
        #         bearing_source = None
        # elif selected_direction_value:
        #     bearing_source = "manual_cardinal"
        # else:
        #     st.caption("Compass heading was not available. Please select an approximate direction.")
        #     bearing_source = None
        selected_direction_value = None
        selected_heading = None
        bearing_source = None

        st.caption("Click the map to update the location.")
        _map = folium.Map(location=[selected_lat, selected_lon], zoom_start=16 if (selected_lat or selected_lon) else 5)
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

        metadata_preview = build_metadata_preview(
            upload_name=upload_name,
            selected_datetime=selected_datetime,
            selected_lat=selected_lat,
            selected_lon=selected_lon,
            selected_heading=selected_heading,
            selected_direction_value=selected_direction_value,
            bearing_source=bearing_source,
            observer_name=st.session_state.get("observer_name_saved", ""),
            top_3_predictions=top_3_predictions,
            selected_habitat_code=selected_habitat_code,
            selected_habitat_name=selected_habitat_name,
            comment=comment,
        )

        st.caption("Please review the Terms and Conditions before submitting.")

        already_uploaded = st.session_state.get("bucket_uploaded_for") == image_id
        if already_uploaded:
            st.success("Image uploaded successfully")
            if st.button("Identify a new habitat", type="primary"):
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


## Render expandable raw API response for diagnostics.
def render_api_response(data):
    """Render raw API response in a collapsible panel for debugging."""
    st.write("## API Response")
    with st.expander("Show response data"):
        st.code(json.dumps(data, indent=2), language="json")


# Warmup function this runs in a separate thread to avoid blocking the main app and to ensure the API is ready (model loaded from cache) when the user makes a request
## Warm up backend model endpoint on app start.
def warmup():
    print("Warming up the API...")
    resp = requests.get(warm_up_url, headers=headers)
    print("Warmup response:", resp.status_code)
## Start warmup in a background thread once per session.
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

    /* ── Submission panel ────────────────────────────────────────────── */
    .st-key-upload_panel,
    .st-key-predictions_panel {
        background-color: #ffffff;
        border: 1px solid #ffffff;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    h1,
    h2 {
        text-align: center;
    }

    div.stButton > button {
        width: 100%;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("AI-Hab is a habitat classification model developed by the [Laboratory of Vision Engineering](https://www.visioneng.org.uk/) at the [University of Lincoln](https://www.lincoln.ac.uk/), and the [UK Centre for Ecology & Hydrology](https://www.ceh.ac.uk/). It is based on the [UKHab](https://www.ukhab.org/) Habitat Classification system and uses computer vision to classify habitats from images. The model is trained on images from the [UKCEH Contryside Survey](https://www.ceh.ac.uk/our-science/projects/countryside-survey).") 

# Take photo or upload
st.write("## Capture or Upload Image")
widget_run = st.session_state.get("widget_run", 0)
camera_img = st.camera_input("Take a photo of the habitat", key=f"camera_{widget_run}")
uploaded_img = st.file_uploader("Or upload a photo", type=["png", "jpg", "jpeg"], key=f"uploader_{widget_run}")
img = camera_img or uploaded_img
location = get_geolocation()

if img:
    image_bytes = img.getvalue()
    file_ext = get_image_file_ext(img)
    image_id = build_image_id(img, image_bytes)

    cached_image_id = st.session_state.get("prediction_image_id")
    if cached_image_id != image_id:
        upload_name = f"habitat_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{file_ext}"
        default_submission_datetime = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        captured_lat, captured_lon = get_location_coords(location)
        # Temporarily disable compass heading capture.
        # captured_heading = get_compass_heading(
        #     measurement_key=f"compass_{widget_run}_{hashlib.sha256(image_bytes).hexdigest()[:10]}"
        # )
        captured_heading = None

        # Send to API only for new images.
        with st.spinner("Analyzing habitat..."):
            data = request_prediction(upload_name, image_bytes, file_ext, captured_lat, captured_lon)

        cache_prediction_state(
            image_id=image_id,
            data=data,
            upload_name=upload_name,
            image_bytes=image_bytes,
            default_submission_datetime=default_submission_datetime,
            captured_lat=captured_lat,
            captured_lon=captured_lon,
            captured_heading=captured_heading,
        )

    data = st.session_state.get("prediction_data")
    upload_name = st.session_state.get("prediction_upload_name")
    cached_bytes = st.session_state.get("prediction_image_bytes")
    prediction_lat = st.session_state.get("prediction_lat")
    prediction_lon = st.session_state.get("prediction_lon")
    prediction_heading = st.session_state.get("prediction_heading")

    if not data:
        st.error("Prediction failed: no response data returned.")
        st.stop()

    # Display results
    predictions = data["results"]["ukhab"]

    if uploaded_img is not None:
        st.image(uploaded_img, caption="Uploaded image")

    render_predictions_panel(predictions, data["inference_time_ms"])
    render_submission_panel(
        image_id=image_id,
        predictions=predictions,
        upload_name=upload_name,
        cached_bytes=cached_bytes,
        prediction_lat=prediction_lat,
        prediction_lon=prediction_lon,
        prediction_heading=prediction_heading,
    )

    render_api_response(data)


# Load licence markdown (cached)
## Load Terms and Conditions markdown content.
@st.cache_data
def load_licence():
    with open("static/licence/licence.md", "r", encoding="utf-8") as f:
        return f.read()


# Define a dialog using the new decorator
## Render Terms and Conditions dialog content.
@st.dialog("Terms and Conditions for AI-Hab API access")
def licence_dialog():
    st.markdown(load_licence())


with st.container(key="app_footer"):
    col1, col2 = st.columns(2)
    with col1:
        st.image("static/img/UKCEH.png")
    with col2:
        st.image("static/img/University-of-Lincoln.png")
    st.markdown("Read the preprint: [Habitat Classification from Ground-Level Imagery Using Deep Neural Networks](https://arxiv.org/abs/2507.04017).")
    st.markdown("View the code to this demonstrator app on [GitHub](https://github.com/NERC-CEH/aihab-streamlit-demo)")
    if st.button("View Terms and Conditions"):
        licence_dialog()