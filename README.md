# Vietnamese Named Entity Recognition (NER) Project

This project implements, compares, and serves multiple models for Vietnamese Named Entity Recognition. It provides a robust FastAPI backend to serve three different NER models through a single API endpoint.

## Live Demonstration

![API Demonstration](demo.gif)

## Features

*   **Multiple Models:** Implementation of three different NER models:
    *   Conditional Random Fields (CRF)
    *   Bidirectional LSTM (BiLSTM)
    *   BiLSTM-CRF
*   **FastAPI Backend:** A single, unified API to serve all three models.
*   **Real-time Inference:** Low-latency predictions with confidence scores for each model.
*   **CORS Enabled:** Allows for easy integration with web frontends.

## Project Structure

*   `crf/`: Contains the implementation of the Conditional Random Fields (CRF) model.
*   `bi_lstm/`: Contains the implementation of the BiLSTM model.
*   `bi_lstm_crf/`: Contains the implementation of the BiLSTM-CRF model and the shared virtual environment.
*   `backend_api.py`: The FastAPI application that serves the models.

## Setup and Installation

1.  **Activate the shared virtual environment:**

    ```bash
    source bi_lstm_crf/.venv/bin/activate
    ```

2.  **Install dependencies (if not already installed):**

    ```bash
    pip install -r bi_lstm_crf/requirements.txt
    pip install fastapi uvicorn
    ```

## Usage

1.  **Run the API server from the project root:**

    ```bash
    uvicorn backend_api:app --host 0.0.0.0 --port 8000
    ```

2.  **Query the `/predict` endpoint:**

    ```bash
    # Example request using the 'bilstm_crf' model
    curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"text": "Tôi là sinh viên trường Đại học Bách khoa Hà Nội.", "model_type": "bilstm_crf"}'
    ```