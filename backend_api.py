# To run the server:
# uvicorn backend_api:app --host 0.0.0.0 --port 8000 --reload

import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# The __init__.py files in subdirectories allow for direct imports.

# Pydantic Models for Request and Response
class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, example="Tôi là sinh viên trường Đại học Bách khoa Hà Nội.")
    model_type: str = Field(..., example="bilstm_crf", pattern="^(bilstm|bilstm_crf|crf)$")

class NERResponse(BaseModel):
    tokens: List[str]
    tags: List[str]
    confidence_scores: List[float]

# Global dictionary to hold the loaded models
models: Dict[str, Any] = {}

# FastAPI application setup
app = FastAPI(
    title="Vietnamese NER API",
    description="An API to serve three different Vietnamese Named Entity Recognition models.",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],   # Allows all methods
    allow_headers=["*"],   # Allows all headers
)

@app.on_event("startup")
def load_models():
    """Load all NER models into memory on application startup."""
    print("Loading models...")
    try:
        # Import inference classes
        from crf.infer import CRFInference
        from bi_lstm.infer import BiLSTMInference
        from bi_lstm_crf.infer import BiLSTMCRFInference

        # Instantiate and store models
        models["crf"] = CRFInference(model_path='crf/model.crfsuite')
        models["bilstm"] = BiLSTMInference(model_path='bi_lstm/checkpoints/ulstm_ner_20.pt')
        models["bilstm_crf"] = BiLSTMCRFInference(model_path='bi_lstm_crf/checkpoints/lstm_ner_10.pt')

        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        # In a production environment, you might want to handle this more gracefully
        # For now, we'll print the error and continue.
        # The application will fail at runtime if a model is requested but not loaded.
        pass

@app.post("/predict", response_model=NERResponse)
def predict(request: NERRequest):
    """Endpoint to perform NER on a given text using a specified model."""
    # Validate model_type and get the corresponding model
    if request.model_type not in models:
        raise HTTPException(status_code=400, detail=f"Invalid model_type: '{request.model_type}'. Available models are: {list(models.keys())}")

    model = models[request.model_type]

    # Perform prediction
    try:
        if request.model_type == 'crf':
            # CRF model has a different prediction method and output format
            predictions = model.predict_text(request.text)
            tokens = [pred[0] for pred in predictions]
            tags = [pred[1] for pred in predictions]
            confidence_scores = [pred[2] for pred in predictions]
        else:
            # BiLSTM and BiLSTM-CRF models share a similar interface
            predictions = model.predict_text(request.text)
            tokens = [pred[0] for pred in predictions]
            tags = [pred[1] for pred in predictions]
            confidence_scores = [pred[2] for pred in predictions]

        return NERResponse(
            tokens=tokens,
            tags=tags,
            confidence_scores=confidence_scores
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

