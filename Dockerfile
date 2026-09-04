FROM python:3.13-slim

# Dependencias del sistema: OpenCV y Tkinter necesitan estas librerias
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Se copian primero las dependencias para aprovechar la cache de capas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV TF_USE_LEGACY_KERAS=1
ENV TF_CPP_MIN_LOG_LEVEL=2

ENTRYPOINT ["python", "detector_neumonia.py"]
