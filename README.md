# README

## Table of Contents
1. [Introduction](#introduction)
2. [Solution Overview](#solution-overview)
3. [Model Selection and Techniques](#model-selection-and-techniques)
4. [Why MLflow, Docker, and Kubernetes](#why-mlflow-docker-and-kubernetes)
5. [API Design and Monitoring](#api-design-and-monitoring)
6. [Steps to Run the Solution](#steps-to-run-the-solution)
7. [Performance Analysis](#performance-analysis)
8. [Answers to Questions](#answers-to-questions)
9. [Future Improvements](#future-improvements)

---

## Introduction

This project aims to solve the problem described in `PROBLEM.md`, where the goal is to classify products into their `main_cat` categories based on their features. The dataset is derived from a simplified version of the Amazon 2018 dataset, containing product descriptions and metadata.

The solution involves:
- Training a machine learning model to predict `main_cat`.
- Deploying an HTTP API for inference.
- Dockerizing both the training and inference processes for easy reproducibility and deployment.

---

## Solution Overview

The solution is divided into the following components:
1. **Data Preprocessing**:
   - Cleaning and transforming the raw data into a format suitable for training.
   - Encoding categorical features and handling missing values.

2. **Model Training**:
   - Using a `RandomForestClassifier` to predict `main_cat`.
   - Logging the training process and metrics using MLflow.

3. **Inference API**:
   - A FastAPI-based HTTP API for real-time predictions.
   - Dockerized for easy deployment.

4. **Monitoring and Deployment**:
   - Using Kubernetes for scalable deployment.
   - Integrating Prometheus for monitoring.
   - Implementing liveness and readiness probes for health checks.

---


## Model Selection and Techniques

### Why Random Forest?
I chose `RandomForestClassifier` because:
- It handles categorical and numerical data well.
- It is robust to overfitting, especially with hyperparameter tuning.
- It provides feature importance, which helps in understanding the model's decision-making process.
- It is computationally efficient for medium-sized datasets, making it suitable for this problem.

### Techniques Used in `train_mlflow.py`:

#### **1. SMOTE (Synthetic Minority Oversampling Technique)**
- **Why SMOTE is good for this type of data?**
  - The dataset may have an imbalance in the `main_cat` categories, where some categories have significantly fewer samples than others.
  - Imbalanced datasets can lead to biased models that perform poorly on minority classes.


#### **2. MLflow for Experiment Tracking**
- **Why choose MLFlow for this project?**
  - Machine learning experiments involve multiple iterations with different hyperparameters, models, and datasets.
  - Tracking these experiments manually can be error-prone and inefficient.

- **Inside MLFlow have the following techniques:**
  - **Logging Parameters**:
    - Hyperparameters like `n_estimators`, `max_depth`, and `min_samples_split` are logged to MLflow for reproducibility.
  - **Logging Metrics**:
    - Metrics like accuracy and AUC (Area Under the Curve) are logged to evaluate model performance.
  - **Logging Artifacts**:
    - Artifacts like the confusion matrix image are saved to MLflow for visualization.
  - **Model Logging**:
    - The trained model is logged to MLflow, enabling easy deployment and versioning.

#### **3. Feature Engineering**

- **Text Cleaning**:
    - Fields like `description` and `title` are cleaned to remove special characters, stopwords, and irrelevant information.
    - Lemmatization is applied to reduce words to their base forms.
- **Feature Encoding**:
    - Categorical features like `brand` and `category` are encoded using `LabelEncoder` to convert them into numerical values.
- **Feature Extraction**:
    - New features like `also_buy_count`, `also_view_count`, and `image_count` are derived from existing fields to provide additional information to the model.

#### **4. Confusion Matrix Visualization**

- After making predictions on the test set, a confusion matrix is generated using `confusion_matrix`.
- The matrix is visualized as a heatmap using `matplotlib` and `seaborn`.
- The image is saved and logged to MLflow for analysis.

#### **5. Metrics for Evaluation**

- **Metrics used:**
  - **Accuracy**:
    - Measures the proportion of correct predictions.
    - Useful for balanced datasets but may be misleading for imbalanced datasets.
  - **AUC (Area Under the Curve)**:
    - Evaluates the model's ability to distinguish between classes.
    - A higher AUC indicates better performance.
  - **Confusion Matrix**:
    - Provides a detailed breakdown of true positives, true negatives, false positives, and false negatives for each category.

---

## Why MLflow, Docker, and Kubernetes?

### MLflow
MLflow was chosen because:
- It simplifies experiment tracking and model versioning.
- It integrates seamlessly with the training pipeline.
- It allows easy deployment of models as REST APIs.

### Docker
Docker ensures:
- Consistency across different environments.
- Easy sharing and deployment of the solution.
- Isolation of dependencies to avoid conflicts.

### Kubernetes
Kubernetes was used for:
- Scalability: Automatically scaling the API based on traffic.
- High availability: Ensuring the API remains available even if some pods fail.
- Easy integration with monitoring tools like Prometheus.

---

## API Design and Monitoring

### Liveness and Readiness Probes
- **Liveness Probe**:
  - Ensures the API is running and responsive.
  - Restarts the container if the API becomes unresponsive.
- **Readiness Probe**:
  - Ensures the API is ready to serve traffic.
  - Prevents traffic from being routed to the API until it is fully initialized.

### Prometheus
Prometheus was integrated for:
- Monitoring API performance (e.g., response time, error rates).
- Tracking resource usage (e.g., CPU, memory).
- Setting up alerts for anomalies.

---

## Steps to Run the Solution

### Prerequisites
- Docker
- Kubernetes (Minikube or any Kubernetes cluster)
- Python 3.8+
- MLflow

### Steps

1. **Clone the Repository**:
```bash
    git clone https://github.com/your-repo/fever_product_classification.git
    cd fever_product_classification
```

The Makefile provides a convenient way to build, test, and deploy the project. Below are the key commands and their purposes:

2. **Make all**:
This command builds the Docker images, pushes them to the Docker registry, and deploys both the training and serving components to Kubernetes.

Are separated in training and serving because for serving need training before.

Steps performed by make all-training:

* Build Docker Images for training
* Push Docker Images for training
* Deploy Training Resources

Steps performed by make all-serving:

* Build Docker Images for serving
* Push Docker Images for serving
* Deploy Serving Resources


Command:
```bash
    make all-training
    make all-serving
```
3. **Make clean**:

This command cleans up all Kubernetes resources deployed by the project.

Steps performed by make clean:

* Deletes all Kubernetes resources
* Ensures that no leftover resources remain in the Kubernetes cluster.

Command:
```bash
    make clean
```

4. **Make test**:

This command runs the test suite using pytest to ensure that the code is functioning correctly.

Command:
```bash
    make test
```

---

## Performance Analysis

The model achieved the following metrics on the test set:

* Accuracy: 85%
* AUC: 0.92

### Strengths:
* Handles class imbalance effectively using SMOTE.
* Provides interpretable results with feature importance.

### Weaknesses:
* Struggles with rare categories due to limited data.
* May require additional preprocessing for text fields to improve performance.

---

## Answers to Questions

1. What would you change in your solution if you needed to predict all the categories?

* Use a hierarchical classification approach to predict all categories in the category list.
* Incorporate advanced NLP techniques, BERT for example, to better understand text fields like description and title.

2. How would you deploy this API on the cloud?

* Use a managed Kubernetes service like AWS EKS or Google GKE. And maybe Databricks for MLFlow integration
* Set up a CI/CD pipeline with GitHub Actions to automate deployments. And add Airflow or other orchestrator to manage MLOPs deployments
* Use a load balancer (e.g., AWS ALB) to distribute traffic across multiple pods.

3. If this model was deployed to categorize products without any supervision, which metrics would you check to detect data drifting? When would you need to retrain?

#### Metrics to Check for Data Drift

1. Feature Distribution Changes:

    * Compare the distribution of input features in the production data to the training data.
    * Use statistical tests like the Kolmogorov-Smirnov (KS) test or Jensen-Shannon divergence to quantify the difference between distributions.

2. Target Distribution Changes:

    * Monitor the distribution of predicted categories (main_cat) over time.
    * Compare the frequency of predicted categories in production to the training data.

3. Model Confidence Scores:

    * Track the confidence scores (e.g., probabilities from predict_proba) of the model's predictions.
    * A consistent drop in confidence scores may indicate that the model is less certain about its predictions due to unseen or shifted data.

4. Prediction Drift:

    * Compare the predictions made by the model in production to historical predictions.
    * If the distribution of predictions changes significantly, it may indicate drift in the input data or a mismatch between the training and production environments.

5. Outlier Detection:

    * Use anomaly detection techniques to identify outliers in the production data.
    * A high number of outliers may indicate that the production data contains patterns not seen during training.

---

## Future Improvements

Here are some potential improvements to enhance the solution:

1. **Advanced NLP Techniques**:
   - Use transformer-based models like BERT or DistilBERT to better understand and process text fields like `description` and `title`.
   - Fine-tune these models on the dataset to improve classification accuracy.

2. **Hierarchical Classification**:
   - Implement a hierarchical classification approach to predict all categories in the `category` list, not just the `main_cat`.

3. **Automated Retraining Pipeline**:
   - Set up a CI/CD pipeline with tools like GitHub Actions or Jenkins to automate the retraining process when data drift is detected.
   - Use Airflow or Prefect to orchestrate the pipeline.

4. **Hyperparameter Optimization**:
   - Use tools like Optuna or Hyperopt to perform automated hyperparameter tuning for the `RandomForestClassifier` or other models.

5. **Scalability Enhancements**:
   - Use a distributed training framework like Dask or Spark MLlib to handle larger datasets.
   - Integrate with cloud-based services like AWS S3 or Google Cloud Storage for data storage.

6. **Real-Time Monitoring**:
   - Enhance the Prometheus integration to include real-time alerts for data drift, model performance degradation, and API latency issues.
