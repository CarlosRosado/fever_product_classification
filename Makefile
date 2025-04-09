# Variables
DOCKER_USERNAME = carlosrosado
PROJECT_NAME = fever_ml
SERVING_IMAGE = serving-fever-image
TRAINING_IMAGE = training-fever-image
TAG = latest
K8S_DIR = ./kubernetes


# Default target
.PHONY: all-training
all-training: build-training push-training deploy-train

# Default target
.PHONY: all-serving
all-serving: build-serving push-serving deploy-serving

# Build Docker images
.PHONY: build-training
build-training:
	@echo "Building Docker images..."
	docker build -t $(DOCKER_USERNAME)/$(TRAINING_IMAGE):$(TAG) -f docker_training/Dockerfile .

# Push Docker images to the registry
.PHONY: push-training
push-training:
	@echo "Pushing Docker images to the registry..."
	docker push $(DOCKER_USERNAME)/$(TRAINING_IMAGE):$(TAG)

# Build Docker images
.PHONY: build-serving
build-serving:
	@echo "Building Docker images..."
	docker build -t $(DOCKER_USERNAME)/$(SERVING_IMAGE):$(TAG) -f docker_serving/Dockerfile .

# Push Docker images to the registry
.PHONY: push-serving
push-serving:
	@echo "Pushing Docker images to the registry..."
	docker push $(DOCKER_USERNAME)/$(SERVING_IMAGE):$(TAG)

# Deploy training to Kubernetes
.PHONY: deploy-train
deploy-train:
	@echo "Deploying training resources to Kubernetes..."
	kubectl apply -f $(K8S_DIR)/configmap.yaml
	kubectl apply -f $(K8S_DIR)/mlflow-deployment.yaml
	kubectl apply -f $(K8S_DIR)/mlflow-service.yaml
	kubectl apply -f $(K8S_DIR)/training-deployment.yaml
	kubectl apply -f $(K8S_DIR)/pvc.yaml

# Deploy serving to Kubernetes
.PHONY: deploy-serving
deploy-serving:
	@echo "Deploying serving resources to Kubernetes..."
	kubectl apply -f $(K8S_DIR)/serving-deployment.yaml
	kubectl apply -f $(K8S_DIR)/serving-service.yaml

# Clean up Kubernetes resources
.PHONY: clean
clean:
	@echo "Cleaning up all Kubernetes resources..."
	kubectl delete -f $(K8S_DIR)/configmap.yaml || true
	kubectl delete -f $(K8S_DIR)/mlflow-deployment.yaml || true
	kubectl delete -f $(K8S_DIR)/mlflow-service.yaml || true
	kubectl delete -f $(K8S_DIR)/serving-deployment.yaml || true
	kubectl delete -f $(K8S_DIR)/serving-service.yaml || true
	kubectl delete -f $(K8S_DIR)/training-deployment.yaml || true
	kubectl delete -f $(K8S_DIR)/pvc.yaml || true
	kubectl delete job ifs-training-job || true

.PHONY: test
test:
	@echo "Running tests..."
	pytest tests --disable-warnings -q
	@echo "Tests completed."

# Rebuild and redeploy everything
.PHONY: rebuild
rebuild: clean all
	@echo "Rebuilt and redeployed everything."




