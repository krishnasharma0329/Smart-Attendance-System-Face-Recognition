#!/bin/bash

echo "🚀 Starting Smart Attendance System with Docker..."

# Allow Docker to access your display (for GUI)
xhost +local:docker

# Create required files if they don't exist
touch attendance.db
mkdir -p dataset
touch trainer.yml

# Build and run
docker-compose up --build

# Clean up display access when done
xhost -local:docker
