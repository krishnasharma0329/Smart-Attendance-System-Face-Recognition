FROM python:3.11-slim

# Install system dependencies for OpenCV + Tkinter + camera
RUN apt-get update && apt-get install -y \
    python3-tk \
    tk-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    libx11-dev \
    x11-apps \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY main.py .

# Create necessary directories
RUN mkdir -p dataset

# Set display environment variable for GUI
ENV DISPLAY=:0

CMD ["python3", "main.py"]
