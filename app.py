import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import os
import hashlib
import html
import re
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
    "u1 - Built-up areas and gardens",
    "w1 - Broadleaved and mixed woodland",
    "w2 - Coniferous Woodland",
    "sea - Sea",
    "c1 - Arable and Horticulture",
    "g4 - Modified Grassland",
    "g3 - Neutral Grassland",
    "g2 - Calcareous Grassland",
    "g1 - Acid Grassland",
    "g1c - Bracken",
    "h1 - Dwarf Shrub Heath",
    "f2 - Fen marsh and swamp",
    "f1 - Bog",
    "t1 - Littoral Rock",
    "t2 - Littoral Sediment",
    "montane - Montane",
    "r1 - Standing Open Waters and Canals",
    "s1 - Inland Rock",
    "s2 - Supralittoral Rock",
    "s3 - Supralittoral Sediment",
]

DEFAULT_MAP_LAT = 54.5
DEFAULT_MAP_LON = -3.0


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
    preserved_browser_lat = st.session_state.get("browser_lat")
    preserved_browser_lon = st.session_state.get("browser_lon")
    preserved_browser_accuracy = st.session_state.get("browser_accuracy")
    new_widget_run = st.session_state.get("widget_run", 0) + 1
    st.session_state.clear()
    if preserved_browser_lat is not None and preserved_browser_lon is not None:
        st.session_state["browser_lat"] = preserved_browser_lat
        st.session_state["browser_lon"] = preserved_browser_lon
    if preserved_browser_accuracy is not None:
        st.session_state["browser_accuracy"] = preserved_browser_accuracy
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
    """Extract lat/lon/accuracy if available from browser geolocation payload."""
    if not location or "coords" not in location:
        return None, None, None
    coords = location["coords"]
    return coords.get("latitude"), coords.get("longitude"), coords.get("accuracy")


## Read and cache browser location for reuse across reruns.
def sync_browser_location(key_hint="default", force_refresh=False):
    """Store the latest successful browser geolocation in session state."""
    nonce_key = f"geo_request_nonce_{key_hint}"
    if force_refresh:
        st.session_state[nonce_key] = st.session_state.get(nonce_key, 0) + 1
    nonce = st.session_state.get(nonce_key, 0)
    location = get_geolocation(component_key=f"getLocation_{key_hint}_{nonce}")
    lat, lon, accuracy = get_location_coords(location)
    if lat is not None and lon is not None:
        st.session_state["browser_lat"] = float(lat)
        st.session_state["browser_lon"] = float(lon)
    if accuracy is not None:
        st.session_state["browser_accuracy"] = float(accuracy)
    return (
        st.session_state.get("browser_lat"),
        st.session_state.get("browser_lon"),
        st.session_state.get("browser_accuracy"),
    )


## Read observer name from localStorage for iframe-safe persistence fallback.
def get_observer_name_from_local_storage():
    """Return observer name persisted in browser localStorage.

    Returns None if the JS hasn't resolved yet (first render), an empty
    string if localStorage has no value, or the stored name string.
    """
    try:
        stored_name = streamlit_js_eval(
            js_expressions="window.localStorage.getItem('observer_name') || ''",
            key="observer_name_local_storage_read",
        )
    except Exception:
        return ""

    if stored_name is None:
        # JS result not available yet on this render pass.
        return None
    return str(stored_name).strip()


## Persist observer name to localStorage.
def set_observer_name_to_local_storage(observer_name):
    """Store observer name in browser localStorage."""
    safe_name = json.dumps(observer_name)
    try:
        streamlit_js_eval(
            js_expressions=f"window.localStorage.setItem('observer_name', {safe_name}); true;",
            key=f"observer_name_local_storage_write_{hashlib.sha1(observer_name.encode('utf-8')).hexdigest()[:10]}",
        )
    except Exception:
        # localStorage can fail in strict privacy modes; ignore and keep session value.
        pass


## Load UKHab taxonomy data for sidebar guidance.
@st.cache_data
def load_ukhab_data():
    """Load UKHab data from the bundled JSON file, or return None on failure."""
    json_path = "data/ukhab.json"
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


## Build dictionary of taxonomy nodes keyed by habitat id.
def build_ukhab_nodes(ukhab_data):
    """Extract non-metadata node records from UKHab JSON."""
    return {
        node_id: node
        for node_id, node in ukhab_data.items()
        if node_id != "_metadata" and isinstance(node, dict) and node.get("id")
    }


## Identify top-level roots by finding nodes never referenced as children.
def get_ukhab_roots(nodes):
    """Return root node ids for rendering the hierarchy."""
    children = {
        child_id
        for node in nodes.values()
        for child_id in node.get("children", [])
        if child_id in nodes
    }
    roots = [node_id for node_id in nodes if node_id not in children]
    return sorted(roots, key=lambda nid: (nodes[nid].get("level", 0), nodes[nid].get("name", nid)))


## Return whether a single node matches the search query text.
def ukhab_node_matches_query(node, search_query):
    """Match query against id/name/definition/inclusions/exclusions fields."""
    if not search_query:
        return True

    parts = [
        str(node.get("id", "")),
        str(node.get("name", "")),
        str(node.get("definition", "")),
    ]
    parts.extend(str(item) for item in (node.get("inclusions") or []))
    parts.extend(str(item) for item in (node.get("exclusions") or []))
    return search_query in " ".join(parts).lower()


## Highlight search terms within visible sidebar text.
def highlight_ukhab_text(text, search_query):
    """Safely wrap matching query terms in a mark tag for sidebar rendering."""
    raw_text = str(text or "")
    if not search_query:
        return html.escape(raw_text)

    tokens = [token for token in re.split(r"\s+", search_query.strip()) if token]
    if not tokens:
        return html.escape(raw_text)

    pattern = re.compile("(" + "|".join(re.escape(token) for token in tokens) + ")", re.IGNORECASE)
    highlighted_parts = []
    last_end = 0

    for match in pattern.finditer(raw_text):
        highlighted_parts.append(html.escape(raw_text[last_end:match.start()]))
        highlighted_parts.append(f"<mark>{html.escape(match.group(0))}</mark>")
        last_end = match.end()

    highlighted_parts.append(html.escape(raw_text[last_end:]))
    return "".join(highlighted_parts)


## Return True when node or any descendant matches the search query.
def ukhab_branch_matches_query(node_id, nodes, search_query, memo, ancestry=None):
    """Check whether a branch should be shown for the active search query."""
    if node_id in memo:
        return memo[node_id]
    if node_id not in nodes:
        memo[node_id] = False
        return False

    ancestry = ancestry or set()
    if node_id in ancestry:
        memo[node_id] = False
        return False

    node = nodes[node_id]
    if ukhab_node_matches_query(node, search_query):
        memo[node_id] = True
        return True

    next_ancestry = set(ancestry)
    next_ancestry.add(node_id)
    for child_id in node.get("children", []):
        if ukhab_branch_matches_query(child_id, nodes, search_query, memo, ancestry=next_ancestry):
            memo[node_id] = True
            return True

    memo[node_id] = False
    return False


## Render one taxonomy node and recurse into children in nested expanders.
def render_ukhab_node(node_id, nodes, search_query="", match_cache=None, ancestry=None):
    """Render a UKHab node in the sidebar as an expandable section."""
    if node_id not in nodes:
        return

    if search_query:
        match_cache = match_cache or {}
        if not ukhab_branch_matches_query(node_id, nodes, search_query, match_cache):
            return

    ancestry = ancestry or set()
    if node_id in ancestry:
        return

    node = nodes[node_id]
    label = f"{node.get('id', node_id)} - {node.get('name', 'Unnamed habitat')}"
    auto_expand = bool(search_query and ukhab_branch_matches_query(node_id, nodes, search_query, match_cache))

    with st.expander(label, expanded=auto_expand):
        level = node.get("level")
        if level is not None:
            st.caption(f"Level {level}")

        definition = (node.get("definition") or "").strip()
        if definition:
            st.markdown(highlight_ukhab_text(definition, search_query), unsafe_allow_html=True)

        inclusions = node.get("inclusions") or []
        if inclusions:
            with st.expander("✅ Inclusions", expanded=False):
                for item in inclusions:
                    st.markdown(
                        f"- {highlight_ukhab_text(item, search_query)}",
                        unsafe_allow_html=True,
                    )

        exclusions = node.get("exclusions") or []
        if exclusions:
            with st.expander("❌ Exclusions", expanded=False):
                for item in exclusions:
                    st.markdown(
                        f"- {highlight_ukhab_text(item, search_query)}",
                        unsafe_allow_html=True,
                    )

        children = [child for child in node.get("children", []) if child in nodes]
        if children:
            if search_query:
                children = [
                    child
                    for child in children
                    if ukhab_branch_matches_query(child, nodes, search_query, match_cache)
                ]

            st.caption("Subtypes")
            next_ancestry = set(ancestry)
            next_ancestry.add(node_id)
            for child_id in children:
                render_ukhab_node(
                    child_id,
                    nodes,
                    search_query=search_query,
                    match_cache=match_cache,
                    ancestry=next_ancestry,
                )


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
    captured_accuracy,
):
    """Persist prediction and submission defaults in session state."""
    resolved_lat = captured_lat if captured_lat is not None else st.session_state.get("browser_lat")
    resolved_lon = captured_lon if captured_lon is not None else st.session_state.get("browser_lon")
    resolved_accuracy = (
        captured_accuracy if captured_accuracy is not None else st.session_state.get("browser_accuracy")
    )

    st.session_state["prediction_image_id"] = image_id
    st.session_state["prediction_data"] = data
    st.session_state["prediction_upload_name"] = upload_name
    st.session_state["prediction_image_bytes"] = image_bytes
    st.session_state["prediction_datetime_utc"] = default_submission_datetime.isoformat()
    st.session_state["submission_date"] = default_submission_datetime.date()
    st.session_state["submission_time"] = default_submission_datetime.time().replace(second=0, microsecond=0)
    st.session_state["prediction_lat"] = resolved_lat
    st.session_state["prediction_lon"] = resolved_lon
    st.session_state["prediction_accuracy"] = resolved_accuracy
    if resolved_lat is not None and resolved_lon is not None:
        st.session_state["map_lat"] = float(resolved_lat)
        st.session_state["map_lon"] = float(resolved_lon)
    else:
        st.session_state["map_lat"] = float(st.session_state.get("map_lat", DEFAULT_MAP_LAT))
        st.session_state["map_lon"] = float(st.session_state.get("map_lon", DEFAULT_MAP_LON))
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


## Resolve habitat description text from loaded UKHab nodes.
def get_ukhab_definition(habitat_code, ukhab_nodes=None):
    """Return UKHab definition for a habitat code, or an empty string."""
    if not ukhab_nodes or habitat_code not in ukhab_nodes:
        return ""
    return str(ukhab_nodes[habitat_code].get("definition", "")).strip()


## Render one prediction row with confidence, hierarchy, and UKHab description.
def render_prediction_content(pred, ukhab_nodes=None):
    """Render the shared prediction UI content for each row."""
    badge_color = confidence_badge_color(pred["confidence"])
    st.markdown(
        f"**{pred['code']} - {pred['name']}** "
        f":{badge_color}-badge[Confidence: {pred['confidence']:.2%}]"
    )
    definition = get_ukhab_definition(pred.get("code"), ukhab_nodes=ukhab_nodes)
    if definition:
        st.caption(definition)
    st.write("" + " > ".join([h["name"] for h in pred["primary_habitat_hierarchy"]]))


## Render the predictions card panel and inference timing.
def render_predictions_panel(predictions, inference_time_ms, image_bytes=None, image_caption=None, ukhab_nodes=None):
    """Render prediction cards in a dedicated styled panel."""
    with st.container(key="predictions_panel"):
        for i, pred in enumerate(predictions):
            # if first prediction add 'top prediction' badge
            if i == 0:
                image_col, prediction_col = st.columns([1, 2], vertical_alignment="top")
                with image_col:
                    if image_bytes is not None:
                        st.image(image_bytes, caption=image_caption, use_container_width=True)
                with prediction_col:
                    st.badge("🥇 Top prediction", color="blue")
                    render_prediction_content(pred, ukhab_nodes=ukhab_nodes)
                continue

            if i == 1:
                st.markdown("---")
                st.badge("Other predictions", color="gray")

            render_prediction_content(pred, ukhab_nodes=ukhab_nodes)
        st.caption(f"Inference time: {inference_time_ms/1000} seconds")


## Construct metadata JSON object for bucket upload.
def build_metadata_preview(
    upload_name,
    selected_datetime,
    selected_lat,
    selected_lon,
    selected_accuracy,
    observer_name,
    top_3_predictions,
    selected_habitat_code,
    selected_habitat_name,
    selected_habitat_level_4_code,
    selected_habitat_level_4_name,
    participant_confidence,
    ai_label_agreement,
    ai_disagreement_reason,
    comment,
):
    """Build metadata document uploaded with each submitted image."""
    return {
        "image_file": upload_name,
        "datetime": selected_datetime.isoformat(),
        "lat": selected_lat,
        "long": selected_lon,
        "location_accuracy_m": selected_accuracy,
        "user_name": observer_name,
        "top_3_predictions": top_3_predictions,
        "selected_habitat_code": selected_habitat_code,
        "selected_habitat_name": selected_habitat_name,
        "selected_habitat_level_4_code": selected_habitat_level_4_code,
        "selected_habitat_level_4_name": selected_habitat_level_4_name,
        "participant_confidence": participant_confidence,
        "ai_label_agreement": ai_label_agreement,
        "ai_disagreement_reason": ai_disagreement_reason,
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
    prediction_accuracy,
    ukhab_nodes=None,
):
    """Render user submission controls and perform bucket upload on submit."""
    with st.container(key="upload_panel"):
        #st.write("### What habitat did you observe?")
        st.info("Your image will be used to evaluate and improve the AI-Hab model. By submitting, you confirm you have permission to upload this image and agree to the Terms and Conditions.")

        ls_name = get_observer_name_from_local_storage()
        # ls_name is None on the first render pass (JS not yet resolved).
        # Do NOT call st.rerun() here — it causes a visible freeze after the modal
        # closes because it triggers a second full rerender in quick succession.
        # Instead seed the widget from localStorage, falling back to "" until it
        # resolves, then let the next natural rerun populate the real value.
        if "observer_name_input" not in st.session_state:
            st.session_state["observer_name_input"] = ls_name if ls_name is not None else ""
        elif ls_name and not st.session_state["observer_name_input"]:
            # localStorage resolved on a later pass and the field is still blank.
            st.session_state["observer_name_input"] = ls_name

        observer_name = st.text_input(
            "Observer full name (will be saved for each new photo you add in this session)",
            key="observer_name_input",
            help="Set once per session. It will be reused for all subsequent observations.",
        )
        normalized_observer_name = observer_name.strip()
        if normalized_observer_name != (ls_name or ""):
            set_observer_name_to_local_storage(normalized_observer_name)

        top_3_predictions = extract_top_predictions(predictions)

        default_habitat = f"{predictions[0]['code']} - {predictions[0]['name']}"
        habitat_options = HABITAT_OPTIONS.copy()
        if default_habitat not in habitat_options:
            habitat_options.insert(0, default_habitat)

        selected_habitat = st.selectbox(
            "Select habitat label (UKHab level 3)",
            options=habitat_options,
            index=habitat_options.index(default_habitat),
            help="Defaults to the top AI prediction. You can choose another habitat from the full list.",
        )
        ai_label_agreement = selected_habitat == default_habitat
        if ai_label_agreement:
            st.caption("👍 Your current selection agrees with the AI's top prediction.")
        else:
            st.caption("Your current selection differs from the AI's top prediction.")

        # Extract Level 3 code for Level 4 lookup
        selected_habitat_parts = selected_habitat.split(" - ", 1)
        selected_habitat_code_l3 = selected_habitat_parts[0]
        selected_habitat_level_4_code = None
        selected_habitat_level_4_name = None

        # Optional Level 4 subspecification (only if UKHab data available)
        if ukhab_nodes and selected_habitat_code_l3 in ukhab_nodes:
            level_3_node = ukhab_nodes[selected_habitat_code_l3]
            level_4_children = level_3_node.get("children", [])
            # Filter to only Level 4 nodes
            level_4_options = [
                child_id for child_id in level_4_children
                if child_id in ukhab_nodes and ukhab_nodes[child_id].get("level") == 4
            ]
            if level_4_options:
                # Build display labels for Level 4 options
                level_4_labels = {}
                for child_id in level_4_options:
                    child_node = ukhab_nodes[child_id]
                    label = f"{child_id} - {child_node.get('name', 'Unknown')}"
                    level_4_labels[label] = child_id

                level_4_selected = st.selectbox(
                    "Refine to UKHab level 4 (optional)",
                    options=["(None)"] + list(level_4_labels.keys()),
                    index=0,
                    help="For confident users: optional more specific habitat classification.",
                    key=f"level_4_select_{image_id}",
                )
                if level_4_selected != "(None)":
                    selected_habitat_level_4_code = level_4_labels[level_4_selected]
                    selected_habitat_level_4_name = ukhab_nodes[selected_habitat_level_4_code].get("name", "")

        participant_confidence = st.select_slider(
            "How confident are you in your selected habitat label?",
            options=[1, 2, 3, 4, 5],
            value=3,
            key=f"participant_confidence_{image_id}",
            help="1 = not confident, 5 = very confident.",
        )

        ai_disagreement_reason = None

        comment = st.text_area("Comment", placeholder="Add an optional note about this habitat observation")

        st.write("---")

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
            )
        with lon_col:
            selected_lon = st.number_input(
                "Longitude",
                value=float(st.session_state.get("map_lon", prediction_lon if prediction_lon is not None else 0.0)),
                min_value=-180.0,
                max_value=180.0,
                format="%.6f",
            )

        # Keep map marker position in sync with any manual coordinate edits.
        st.session_state["map_lat"] = float(selected_lat)
        st.session_state["map_lon"] = float(selected_lon)

        map_hint_col, map_refresh_col = st.columns([3, 2])
        with map_hint_col:
            st.caption("Click the map to update the location.")
        with map_refresh_col:
            if st.button(
                "Use device location",
                key=f"refresh_location_{image_id}",
                use_container_width=True,
            ):
                refreshed_lat, refreshed_lon, refreshed_accuracy = sync_browser_location(
                    key_hint=f"refresh_{image_id}",
                    force_refresh=True,
                )
                if refreshed_lat is not None and refreshed_lon is not None:
                    st.session_state["prediction_lat"] = float(refreshed_lat)
                    st.session_state["prediction_lon"] = float(refreshed_lon)
                    if refreshed_accuracy is not None:
                        st.session_state["prediction_accuracy"] = float(refreshed_accuracy)
                    st.session_state["map_lat"] = float(refreshed_lat)
                    st.session_state["map_lon"] = float(refreshed_lon)
                    st.rerun()
                else:
                    st.warning("Could not retrieve device location. Please check browser permissions.")

        _map = folium.Map(
            location=[selected_lat, selected_lon],
            zoom_start=16 if (selected_lat or selected_lon) else 5,
            control_scale=True,
        )
        folium.CircleMarker(
            location=[selected_lat, selected_lon],
            radius=8,
            color="#d63384",
            weight=2,
            fill=True,
            fill_color="#ff4b4b",
            fill_opacity=0.9,
            tooltip="Observation location",
        ).add_to(_map)
        map_result = st_folium(
            _map,
            height=250,
            width="100%",
            returned_objects=["last_clicked"],
            key=f"observation_map_{image_id}",
        )
        if map_result and map_result.get("last_clicked"):
            clicked_lat = map_result["last_clicked"]["lat"]
            clicked_lng = map_result["last_clicked"]["lng"]
            if (clicked_lat, clicked_lng) != (st.session_state.get("map_lat"), st.session_state.get("map_lon")):
                st.session_state["map_lat"] = clicked_lat
                st.session_state["map_lon"] = clicked_lng
                st.session_state["prediction_accuracy"] = None
                st.rerun()

        browser_lat = st.session_state.get("browser_lat")
        browser_lon = st.session_state.get("browser_lon")
        browser_accuracy = st.session_state.get("browser_accuracy")
        selected_accuracy = st.session_state.get("prediction_accuracy", prediction_accuracy)
        if browser_lat is not None and browser_lon is not None:
            if abs(selected_lat - float(browser_lat)) > 1e-8 or abs(selected_lon - float(browser_lon)) > 1e-8:
                selected_accuracy = None

        if selected_accuracy is not None:
            st.caption(f"Device geolocation accuracy: +/-{float(selected_accuracy):.1f} m")
        else:
            st.caption("Device geolocation accuracy: unavailable for the current location.")
        selected_habitat_code = selected_habitat_code_l3
        selected_habitat_name = selected_habitat_parts[1] if len(selected_habitat_parts) > 1 else ""

        metadata_preview = build_metadata_preview(
            upload_name=upload_name,
            selected_datetime=selected_datetime,
            selected_lat=selected_lat,
            selected_lon=selected_lon,
            selected_accuracy=selected_accuracy,
            observer_name=normalized_observer_name,
            top_3_predictions=top_3_predictions,
            selected_habitat_code=selected_habitat_code,
            selected_habitat_name=selected_habitat_name,
            selected_habitat_level_4_code=selected_habitat_level_4_code,
            selected_habitat_level_4_name=selected_habitat_level_4_name,
            participant_confidence=participant_confidence,
            ai_label_agreement=ai_label_agreement,
            ai_disagreement_reason=ai_disagreement_reason,
            comment=comment,
        )

        st.caption("Please review the Terms and Conditions before submitting.")

        already_uploaded = st.session_state.get("bucket_uploaded_for") == image_id
        if already_uploaded:
            st.success("Observation submitted successfully. Thank you for your contribution! ")
            if st.button("Identify a new habitat", type="primary"):
                reset_for_new_habitat()
                st.rerun()
        elif st.button("Submit observation", type="primary"):
            if not normalized_observer_name:
                st.error("Please enter your user name before uploading.")
                st.stop()

            with st.spinner("Uploading image and metadata..."):
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
    with st.expander("Show API response data"):
        st.code(json.dumps(data, indent=2), language="json")


## Switch to a specific rendered tab after a rerun.
def switch_to_rendered_tab(tab_index, nonce=0):
    """Use a small frontend script to activate a Streamlit tab by index."""
    components.html(
        f"""
        <div data-tab-switch-nonce="{nonce}" style="display:none"></div>
        <script>
        const clickTab = () => {{
            const parentDoc = window.parent.document;
            const tabs = parentDoc.querySelectorAll('button[role="tab"]');
            if (tabs.length > {tab_index}) {{
                const targetTab = tabs[{tab_index}];
                targetTab.click();

                const anchor = parentDoc.getElementById('results-tabs-anchor');
                if (anchor) {{
                    anchor.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }} else {{
                    targetTab.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }}
                return true;
            }}
            return false;
        }};

        if (!clickTab()) {{
            const interval = setInterval(() => {{
                if (clickTab()) {{
                    clearInterval(interval);
                }}
            }}, 100);
            setTimeout(() => clearInterval(interval), 2000);
        }}
        </script>
        """,
        height=0,
    )


## Open a fresh modal flow for capture or upload tasks.
def open_photo_modal(dialog_fn):
    """Reset the current image workflow and open the requested modal."""
    reset_for_new_habitat()
    dialog_fn()


## Process an image inside the active modal and return results to the main page.
def render_image_workflow(img, source_caption):
    """Analyze an image, cache the result, then close the modal."""
    if not img:
        return

    image_bytes = img.getvalue()
    file_ext = get_image_file_ext(img)
    image_id = build_image_id(img, image_bytes)

    cached_image_id = st.session_state.get("prediction_image_id")
    if cached_image_id != image_id:
        upload_name = f"habitat_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{file_ext}"
        default_submission_datetime = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        captured_lat = st.session_state.get("browser_lat")
        captured_lon = st.session_state.get("browser_lon")
        captured_accuracy = st.session_state.get("browser_accuracy")

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
            captured_accuracy=captured_accuracy,
        )

    data = st.session_state.get("prediction_data")
    if not data:
        st.error("Prediction failed: no response data returned.")
        return

    st.session_state["prediction_source_caption"] = source_caption
    st.rerun()


## Render cached prediction results on the main page.
def render_cached_results():
    """Show the latest analyzed image and its prediction details."""
    data = st.session_state.get("prediction_data")
    image_id = st.session_state.get("prediction_image_id")
    upload_name = st.session_state.get("prediction_upload_name")
    cached_bytes = st.session_state.get("prediction_image_bytes")
    prediction_lat = st.session_state.get("prediction_lat")
    prediction_lon = st.session_state.get("prediction_lon")
    prediction_accuracy = st.session_state.get("prediction_accuracy")
    source_caption = st.session_state.get("prediction_source_caption", "Habitat image")

    if not data or not image_id or not upload_name or not cached_bytes:
        return

    predictions = data["results"]["ukhab"]

    st.markdown("<div id='results-tabs-anchor'></div>", unsafe_allow_html=True)
    predictions_tab, submission_tab = st.tabs(["Predictions", "Submit observation"])
    with predictions_tab:
        render_predictions_panel(
            predictions,
            data["inference_time_ms"],
            image_bytes=cached_bytes,
            image_caption=source_caption,
            ukhab_nodes=ukhab_nodes,
        )
        if st.button(
            "Continue to submit observation",
            key="go_to_submit_tab",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["switch_to_submission_tab"] = True
            st.session_state["tab_switch_nonce"] = st.session_state.get("tab_switch_nonce", 0) + 1
            st.rerun()

    with submission_tab:
        render_submission_panel(
            image_id=image_id,
            predictions=predictions,
            upload_name=upload_name,
            cached_bytes=cached_bytes,
            prediction_lat=prediction_lat,
            prediction_lon=prediction_lon,
            prediction_accuracy=prediction_accuracy,
            ukhab_nodes=ukhab_nodes,
        )

    if st.session_state.pop("switch_to_submission_tab", False):
        switch_to_rendered_tab(1, st.session_state.get("tab_switch_nonce", 0))

    render_api_response(data)

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


st.set_page_config(
    page_title="AI-Hab Identify",
    page_icon="static/img/ai-hab-logo-transparent.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)
start_warmup_thread()

# Load UKHab data at module scope so it's accessible throughout the app
ukhab_data = load_ukhab_data()
ukhab_nodes = None
if isinstance(ukhab_data, dict) and ukhab_data:
    ukhab_nodes = build_ukhab_nodes(ukhab_data)

with st.sidebar:

    st.title("UKHab hierarchy")
    if not isinstance(ukhab_data, dict) or not ukhab_data:
        st.warning("UKHab guidance is currently unavailable because the taxonomy JSON file could not be loaded.")
    else:
        ukhab_roots = get_ukhab_roots(ukhab_nodes)
        metadata = ukhab_data.get("_metadata", {})

        st.markdown("Quick reference for UKHab level 3 and 4 habitats. Access the full [UKHab documentation here](https://ukhab.org/).")
        if metadata.get("version"):
            st.caption(f"Version {metadata['version']}")

        search_query = st.text_input(
            "Search UKHab",
            value="",
            placeholder="Search by code, name, or definition",
            key="ukhab_sidebar_search",
        ).strip().lower()
        match_cache = {}

        visible_root_count = 0
        for root_id in ukhab_roots:
            if search_query and not ukhab_branch_matches_query(root_id, ukhab_nodes, search_query, match_cache):
                continue
            visible_root_count += 1
            render_ukhab_node(root_id, ukhab_nodes, search_query=search_query, match_cache=match_cache)

        if search_query and visible_root_count == 0:
            st.info("No matching habitats found.")

st.markdown(
    """
    <style>

    section.main > div.block-container,
    div[data-testid="stMainBlockContainer"] {
        max-width: 860px !important;
        width: 100%;
        margin: 0 auto !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

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

    div.stButton,
    div.stButton > button {
        width: 100%;
    }

    .st-key-home_actions [data-testid="stHorizontalBlock"] {
        align-items: stretch;
    }

    .st-key-home_actions .stButton > button {
        min-height: 7.5rem;
        font-weight: 700;
        border-radius: 16px;
        padding: 0.75rem 1rem;
    }

    @media (max-width: 900px) {
        .st-key-home_actions [data-testid="stHorizontalBlock"] {
            flex-direction: column;
        }

        .st-key-home_actions [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
    }

    .st-key-home_actions .stButton > button p {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        line-height: 1.2;
        margin: 0;
    }

    .st-key-location_sync {
        height: 0;
        overflow: hidden;
        opacity: 0;
        margin: 0;
        padding: 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)
with st.container(key="location_sync"):
    if st.session_state.get("browser_lat") is None or st.session_state.get("browser_lon") is None:
        sync_browser_location(key_hint="startup")

st.markdown("# AI-Hab *Identify*")
st.markdown("Identify habitats to UK-Hab, powered by AI.")

@st.dialog("Take a habitat photo")
def photo_capture_dialog():
    st.info(
        "**Tips for a good habitat photo**\n"
        "- Hold the camera steady and keep the horizon level.\n"
        "- Include the main habitat clearly, not just a close-up detail.\n"
        "- Avoid blur, heavy shadow, or pointing directly into bright sunlight.\n"
    )

    widget_run = st.session_state.get("widget_run", 0)
    camera_img = st.camera_input("Take a photo of the habitat", key=f"camera_{widget_run}")
    render_image_workflow(camera_img, "Captured image")


@st.dialog("Upload a habitat photo")
def photo_upload_dialog():
    st.info(
        "**Tips for choosing a good photo**\n"
        "- Pick an image where the habitat is the main subject.\n"
        "- Avoid screenshots, collages, or heavily edited images.\n"
        "- Choose a sharp photo with good lighting if possible.\n"
        "- Wider scene photos are usually more useful than extreme close-ups."
    )

    widget_run = st.session_state.get("widget_run", 0)
    uploaded_img = st.file_uploader(
        "Choose an existing habitat image from your device",
        type=["png", "jpg", "jpeg"],
        key=f"uploader_{widget_run}",
    )
    render_image_workflow(uploaded_img, "Uploaded image")

with st.container(key="home_actions"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "📷 Take a photo",
            key="open_camera_modal",
            type="primary",
            use_container_width=True,
        ):
            open_photo_modal(photo_capture_dialog)
    with col2:
        if st.button(
            "🖼️ Upload a photo",
            key="open_upload_modal",
            use_container_width=True,
        ):
            open_photo_modal(photo_upload_dialog)

render_cached_results()


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
    st.write("---")
    st.markdown("AI-Hab is a habitat classification model developed by the [Laboratory of Vision Engineering](https://www.visioneng.org.uk/) at the [University of Lincoln](https://www.lincoln.ac.uk/), and the [UK Centre for Ecology & Hydrology](https://www.ceh.ac.uk/). It is based on the [UKHab](https://www.ukhab.org/) Habitat Classification system and uses computer vision to identify habitats from images. The model is trained on images from the [UKCEH Contryside Survey](https://www.ceh.ac.uk/our-science/projects/countryside-survey).") 
    col1, col2 = st.columns(2)
    with col1:
        st.image("static/img/UKCEH.png")
    with col2:
        st.image("static/img/University-of-Lincoln.png")
    st.markdown("Read the preprint: [Habitat Classification from Ground-Level Imagery Using Deep Neural Networks](https://arxiv.org/abs/2507.04017).")
    st.markdown("View the code to this demonstrator app on [GitHub](https://github.com/NERC-CEH/aihab-streamlit-demo)")

    if st.button("View Terms and Conditions"):
        licence_dialog()