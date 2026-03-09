# Usar una imagen ligera de Python
FROM python:3.8-slim

# Instalar dependencias del sistema para procesamiento de imágenes
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar e instalar requerimientos
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY app/ .

# Exponer el puerto de la API
EXPOSE 80

CMD ["python", "app.py"]