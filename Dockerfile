FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

ARG WANDB_API

WORKDIR /code
COPY . .

# Install Python3.8
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    vim \
    libglu1 libgconf-2-4 libatk1.0-0 libatk-bridge2.0-0 libgdk-pixbuf2.0-0 libgtk-3-0 libgbm-dev libnss3-dev libxss-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN wandb login ${WANDB_API}