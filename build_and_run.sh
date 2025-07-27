#!/bin/bash

# Docker build and run script for WSL2
# Save this as build_and_run.sh and make it executable: chmod +x build_and_run.sh

set -e  # Exit on any error

echo "=== Docker Build and Run Script for Adobe Hackathon ==="

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Check if required directories exist
if [ ! -d "src" ]; then
    echo "❌ src directory not found. Please run from project root."
    exit 1
fi

if [ ! -d "Challenge_1b" ]; then
    echo "❌ Challenge_1b directory not found."
    echo "Please ensure Challenge_1b directory exists with test data."
    exit 1
fi

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t adobe-hackathon:latest . --no-cache

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Docker image built successfully!"

# Check image size
IMAGE_SIZE=$(docker images adobe-hackathon:latest --format "table {{.Size}}" | tail -n 1)
echo "📦 Image size: $IMAGE_SIZE"

# List collections in Challenge_1b
echo "📁 Available collections:"
ls -la Challenge_1b/ | grep "Collection" || echo "No Collection directories found"

# Run the container
echo "🚀 Running the application..."
docker run --rm \
    -v "$(pwd)/Challenge_1b:/app/Challenge_1b" \
    -v "$(pwd)/outputs:/app/outputs" \
    --name adobe-hackathon-run \
    adobe-hackathon:latest

if [ $? -eq 0 ]; then
    echo "✅ Application completed successfully!"
    echo "📄 Check outputs/ directory for results"
else
    echo "❌ Application failed!"
fi

# Optional: Show logs if needed
echo ""
echo "💡 To debug, run:"
echo "docker run -it --rm -v \"\$(pwd)/Challenge_1b:/app/Challenge_1b\" adobe-hackathon:latest /bin/bash"