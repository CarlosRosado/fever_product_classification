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

# Copy the entire project into the container
COPY . /app

# Expose the port for the app
EXPOSE 8002

# Set environment variables for serving
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MODEL_NAME=Fever_Random_Forest
ENV OPENAPI_SPEC_PATH=/app/fever_product_classification/app/prediction-openapi.yaml

# run the FastAPI app
CMD ["uvicorn", "app.serving:app", "--host", "0.0.0.0", "--port", "8002"]