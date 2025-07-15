FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python 3.9
RUN apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3.9-distutils python3.9-venv \
    python3-pip git wget cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    update-alternatives --set python /usr/bin/python3.9

# Install pip for Python 3.9
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Install ALFWorld dependencies
RUN pip install --no-cache-dir ai2thor==2.1.0 networkx==2.3 numpy==1.24.4 \
    opencv-python==4.1.2.30 Pillow==9.5.0 tqdm==4.32.2

# Install required packages
RUN pip install --no-cache-dir \
    google-generativeai \
    pydantic \
    PyYAML \
    beautifulsoup4 \
    requests \
    numpy

WORKDIR /home/alfworld
