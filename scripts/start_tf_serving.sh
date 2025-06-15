#!/bin/bash

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

# Load .env variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^\s*$' | xargs)
else
    echo "[ERROR] .env file not found"
    exit 1
fi

CONTAINER_NAME=tf_serving_container

# Download the tensorflow serving docker image
docker pull tensorflow/serving

# Remove any existing container with same name
docker rm -f $CONTAINER_NAME > /dev/null 2>&1 || true

# Start container in background and detach logs to a file
docker run -d \
  --name $CONTAINER_NAME \
  -p 8501:8501 \
  -v "$(pwd)/${MODEL_DIR}:/${TF_SERVING_MOUNT_PATH}" \
  -v "$(pwd)/${TF_SERVING_CONFIG_PATH}:/models.config" \
  tensorflow/serving \
  --model_config_file=/models.config

echo "TensorFlow Serving container started as $CONTAINER_NAME"
