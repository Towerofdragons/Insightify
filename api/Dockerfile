# syntax=docker/dockerfile:1
FROM python:3.10-slim
WORKDIR /api

# ENV FLASK_APP=app.py
# ENV FLASK_RUN_HOST=0.0.0.0

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Torch first (this layer rarely changes)
# RUN pip install torch


# Install dependencies 
# RUN pip install --no-cache-dir sentence-transformers

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip \ 
  && pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
COPY . .
CMD ["python3", "ETL-Proper.py"]