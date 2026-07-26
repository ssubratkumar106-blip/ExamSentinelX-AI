# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV PORT=5000

# Install system dependencies (required for AI models and WebSockets)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create directories for logs and uploads
RUN mkdir -p captures/evidence reports/generated logs database

# Expose port (Render sets this dynamically, but EXPOSE is good for docs)
EXPOSE 5000

# Start the application with Gunicorn and Eventlet worker for SocketIO
CMD gunicorn --worker-class gthread --threads 4 --timeout 120 -w 1 --bind 0.0.0.0:$PORT run:app
