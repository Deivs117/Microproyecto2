FROM python:3.8-slim

# Mantenemos las dependencias de sistema necesarias para MXNet
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libquadmath0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos e instalamos los requerimientos con la versión de numpy fija
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 80

CMD ["python", "app.py"]