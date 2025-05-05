# Use official Python image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install dependencies
RUN apt-get update && apt-get install -y build-essential libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .

# Expose port for Gunicorn
EXPOSE 8000

CMD ["uvicorn", "DesignAI.asgiaws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/v2q6v1m6:application", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
