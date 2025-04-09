# Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app  

# Set the working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the src directory
COPY ../src /app/src

# Copy the pipeline_mlflow.py script from src/pipeline
COPY ../src/model/train_mlflow.py /app/src/model/train_mlflow.py

# Set environment variables for training
ENV DATA_URL=https://drive.google.com/uc?id=1Zf0Kdby-FHLdNXatMP0AD2nY0h-cjas3
ENV LOCAL_DATA_PATH=/app/data_files/
ENV MODEL_OUTPUT_PATH=/app/models/
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MODEL_NAME=Fever_Random_Forest
ENV OPENAPI_SPEC_PATH=/app/fever_product_classification/app/prediction-openapi.yaml

# run the training script
CMD ["python", "/app/src/model/train_mlflow.py"]