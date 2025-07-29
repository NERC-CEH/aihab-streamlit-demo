# AI-Hab API usage Demonstrator

This is a minimal demonstrator streamlit web application that classifies habitats into **UKHab Level 3 categories** using the AI-Hab computer computer vision model. This app serves as an example of integrating the API into an app.

The API codebase is available here: https://github.com/NERC-CEH/aihab-api

The API is in development and is currently hosted on the UKCEH Posit Connect server, it is only accessible via authentication.

## Features

* **Camera capture or file upload**: Take a photo directly in the app or upload an existing image.
* **API-powered predictions**: Sends the image to a Posit Connect API for habitat classification.
* **Expandable API response**: Inspect full JSON responses directly in the app.

## Requirements

* Python 3.8+
* [Streamlit](https://streamlit.io)
* `requests`
* `python-dotenv`

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

## Running the App

Run the Streamlit app locally:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Environment Variables

* `API_KEY`: Your API key for authenticating with the API.
* `API_URL`: Base URL of the API.

## Project Structure

```
├── app.py                # Main Streamlit app
├── static/
│   ├── img/
│   │   ├── logos etc.
├── .env                  # Environment variables (not committed)
└── requirements.txt      # Python dependencies
```
