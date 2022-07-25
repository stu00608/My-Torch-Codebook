FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

WORKDIR /code
COPY . .

# Install Python3.8
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3-pip \
    python3.8-dev \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3.8 -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python3.8 -m pip install --no-cache-dir -r requirements.txt