from fastapi import FastAPI, HTTPException
import numpy as np
import mlflow.pyfunc
import os
import pandas as pd
from prometheus_client import start_http_server, Summary, Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import yaml
from pydantic import BaseModel, Field

# Set the MLflow Tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

# Initialize FastAPI app
app = FastAPI()

# Create Prometheus metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('request_count', 'Total number of requests')

# Load the model
MODEL_NAME = os.getenv("MODEL_NAME", "Fever_Random_Forest")
MODEL_VERSION = os.getenv("MODEL_VERSION")  # Optional: Specify a version
try:
    if MODEL_VERSION:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    else:
        model_uri = f"models:/{MODEL_NAME}/latest"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model '{MODEL_NAME}' loaded successfully from URI: {model_uri}")
except Exception as e:
    print(f"Error loading model '{MODEL_NAME}': {e}")
    model = None

class Features(BaseModel):
    also_buy_count: int = Field(..., description="Count of also_buy items")
    also_view_count: int = Field(..., description="Count of also_view items")
    asin: str = Field(..., description="ASIN of the product")
    brand: str = Field(..., description="Brand of the product")
    category: str = Field(..., description="Category of the product")
    description: str = Field(..., description="Description of the product")
    feature: str = Field(..., description="Feature of the product")
    image_count: int = Field(..., description="Count of images")
    price: float = Field(..., description="Price of the product")
    title: str = Field(..., description="Title of the product")

class PredictionRequest(BaseModel):
    features: Features

@app.post("/predict")
@REQUEST_TIME.time()
@REQUEST_COUNT.count_exceptions()
@app.post("/predict")
def predict(data: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Cannot make predictions.")

    try:
        # Extract features from Pydantic model
        features = data.features.dict()

        # Preprocess features
        for field in ["asin", "brand", "category", "description", "feature", "title"]:
            if field in features and isinstance(features[field], str):
                features[field] = hash(features[field]) % (10**6)  

        # Convert features to a DataFrame
        features_df = pd.DataFrame([features])

        # Make predictions
        prediction = model.predict(features_df)
        return {"model": MODEL_NAME, "version": MODEL_VERSION or "latest", "prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")

        
@app.get("/")
def read_root():
    """
    Root endpoint for the API.

    Returns:
        dict: A welcome message.
    """
    return {"message": "Welcome to the Fever Product Classification API"}


# Load OpenAPI specification
OPENAPI_SPEC_PATH = os.getenv("OPENAPI_SPEC_PATH", "prediction-openapi.yaml")  # Default to a relative path
try:
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    openapi_spec_path = os.path.join(script_dir, OPENAPI_SPEC_PATH) 
    with open(openapi_spec_path, "r") as f:
        openapi_spec = yaml.safe_load(f)
except FileNotFoundError:
    openapi_spec = None
    print(f"OpenAPI specification file not found at {openapi_spec_path}")

@app.get("/metrics")
def metrics():
    """
    Endpoint for returning the current metrics of the service.

    Returns:
        Response: The current metrics in Prometheus format.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/specifications")
def get_specifications():
    """
    Endpoint for returning the OpenAPI specifications.

    Returns:
        dict: The OpenAPI specifications.
    """
    if openapi_spec:
        return openapi_spec
    else:
        raise HTTPException(status_code=404, detail="OpenAPI specification not found.")

# Start up the server to expose the metrics.
start_http_server(9092)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serving:app", host="0.0.0.0", port=8002)