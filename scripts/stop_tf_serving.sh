#!/bin/bash

CONTAINER_NAME=tf_serving_container

echo "Stopping TensorFlow Serving container $CONTAINER_NAME..."
docker stop $CONTAINER_NAME >/dev/null 2>&1 || echo "Container not running"
docker rm $CONTAINER_NAME
echo "Container stopped and removed"