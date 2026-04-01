---
title: AI-Hab API Usage Demo
pinned: false
sdk: streamlit
---
# AI-Hab API usage Demonstrator

This is a minimal demonstrator streamlit web application that classifies habitats into **UKHab Level 3 categories** using the AI-Hab computer computer vision model. This app serves as an example of integrating the API into an app.

The API codebase is available here: https://github.com/NERC-CEH/aihab-api

The API is in development and is currently hosted on the UKCEH Posit Connect server, it is only accessible via authentication.

## Features

* **Camera capture or file upload**: Take a photo directly in the app or upload an existing image.
* **API-powered predictions**: Sends the image to a Posit Connect API for habitat classification.
* **Manual Hugging Face bucket upload**: After prediction, press a button to upload data to `aihab-uk/habitatimages`.
* **Metadata capture**: Captures datetime, location, top-3 predictions with confidence, selected habitat label, and user comment.
* **Expandable API response**: Inspect full JSON responses directly in the app.

## Requirements

* Python 3.8+
* [Streamlit](https://streamlit.io)
* `requests`
* `python-dotenv`
* `huggingface_hub`

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API details:

   ```bash
   API_KEY=<your-api-key>
   API_URL=<your-api-url>
   ```

4. Add your Hugging Face token in your environment configuration:

   ```bash
   HF_AUTH_TOKEN=<your-huggingface-token>
   ```

## Running the App

Run the Streamlit app locally:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Environment Variables

* `API_KEY`: Your API key for authenticating with the API.
* `API_URL`: Base URL of the API.
* `HF_AUTH_TOKEN`: Hugging Face token used for bucket uploads.
* `HF_BUCKET_PREFIX` (optional): Prefix/folder inside the fixed bucket `aihab-uk/habitatimages`.

## Hugging Face Bucket Upload

After the prediction response is shown, press **Submit image to bucket** to upload to `hf://buckets/aihab-uk/habitatimages`.

Files are organized as:

* `images/<image-file>`
* `metadata/<image-file-stem>.json`

The metadata JSON contains:

* Image file name
* Datetime
* Latitude and longitude (when available)
* Top 3 AI-predicted habitats and confidence values
* Habitat label selected by the user
* User comment

## Project Structure

```
├── app.py                # Main Streamlit app
├── static/
│   ├── img/
│   │   ├── logos etc.
├── .env                  # Environment variables (not committed)
└── requirements.txt      # Python dependencies
```
