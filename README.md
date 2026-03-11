# Microproyecto 2: Orquestación de Contenedores y ML en la Nube
**Estado:** Punto 1 - Infraestructura AKS Completada

## 1. Requisitos Previos
- Azure CLI instalado y autenticado.
- Suscripción Azure for Students activa.
- Registro de proveedores de recursos:
  - `Microsoft.Compute`
  - `Microsoft.ContainerService`

Comandos para la sección de Requisitos Previos
Agrégalos debajo de los puntos correspondientes en tu archivo:

A. Autenticación en Azure
Para iniciar sesión desde tu terminal local (como Warp o Ubuntu):

```bash

az login

```

B. Registro de Proveedores de Recursos
Estos comandos habilitan la capacidad de crear Máquinas Virtuales (Compute) y clústeres administrados (AKS) en tu suscripción:

```bash

# Registrar el proveedor de Cómputo (VMs)
az provider register --namespace Microsoft.Compute

# Registrar el proveedor de Kubernetes (AKS)
az provider register --namespace Microsoft.ContainerService
```

C. Verificación del Registro
Es vital confirmar que el estado sea Registered antes de lanzar el clúster, ya que el proceso de registro puede tardar un par de minutos:

```bash

# Verificar estado de Microsoft.Compute
az provider show -n Microsoft.Compute 
--query registrationState

# Verificar estado de Microsoft.ContainerService
az provider show -n Microsoft.ContainerService --query registrationState
```

## 2. Configuración de Infraestructura (Punto 1)

### Creación del Grupo de Recursos
Se utiliza la región `centralus` por disponibilidad de cuotas para cuentas académicas.
```bash
az group create --name RG_Microproyecto2 --location centralus
```
Implementación del Clúster AKS
Se despliega un clúster de 2 nodos usando el SKU Standard_D2s_v3 (2 vCPUs, 8GiB RAM) para cumplir con los requisitos del proyecto.

```bash

az aks create \
    --resource-group RG_Microproyecto2 \
    --name ClusterIA \
    --location centralus \
    --node-count 2 \
    --node-vm-size Standard_D2s_v3 \
    --generate-ssh-keys
```



Configuración de Acceso
Comandos para vincular la terminal local/Cloud Shell con el clúster:

```bash

az aks get-credentials --resource-group RG_Microproyecto2 --name ClusterIA
```

Verificación de Salud

```bash

kubectl get nodes
```

Resultado esperado: 2 nodos en estado 'Ready'.

---

## Punto 2: Clasificador de Imágenes con ResNet20

Este documento detalla los pasos para desplegar un modelo de Deep Learning en **Azure Kubernetes Service (AKS)**.

### 1. Configuración de Archivos de Manifiesto

Ejecuta estos comandos en la terminal de Azure Cloud Shell para generar automáticamente los archivos de configuración necesarios:

#### A. Crear Manifiesto de Despliegue (`deployment.yaml`)

Este archivo define 2 réplicas del contenedor de IA para asegurar alta disponibilidad.

```bash
cat <<EOF > deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kubermatic-dl-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: image-classifier
  template:
    metadata:
      labels:
        app: image-classifier
    spec:
      containers:
      - name: classifier-container
        image: davids117/image-classifier:v1
        ports:
        - containerPort: 80
EOF

```

#### B. Crear Manifiesto de Servicio (`service.yaml`)

Este archivo expone la aplicación a internet mediante una IP pública de Azure.

```bash
cat <<EOF > service.yaml
apiVersion: v1
kind: Service
metadata:
  name: image-classifier-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 80
  selector:
    app: image-classifier
EOF

```

---

### 2. Comandos de Despliegue

Una vez creados los archivos anteriores, sigue este orden:

```bash
# 1. Vincular la terminal con el clúster de Azure
az aks get-credentials --resource-group RG_Microproyecto2 --name ClusterIA

# 2. Aplicar los manifiestos en el clúster
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# 3. Monitorear hasta obtener la IP Pública (EXTERNAL-IP)
kubectl get service image-classifier-service --watch

```

---

### 3. Prueba de la Aplicación (Client-Side)

Desde la terminal local (donde se encuentre la carpeta de imágenes), ejecutar:

```bash
# Reemplazar <IP_AZURE> con el valor obtenido en el paso anterior
curl.exe -X POST -F "img=@assets/horse.jpg" http://<IP_AZURE>/predict

```

---

### 4. Notas Técnicas para Sustentación

* **Corrección de Imagen:** Se utiliza el repositorio `davids117/image-classifier:v1`.
* **Ajuste de Código:** Se configuró `pretrained=True` para que el contenedor descargue los pesos del modelo automáticamente al iniciar.
* **Infraestructura:** El clúster corre en 2 nodos `Standard_D2s_v3` en la región `centralus`.

---

Aquí tienes la documentación técnica lista para integrar en tu archivo **README.md**. Este bloque cubre desde la creación de la infraestructura hasta la validación de las métricas de tu servicio de detección de imágenes (AI vs Real).

---

# Punto 3: Despliegue de Aplicación Propia y Monitoreo

Este apartado detalla el proceso para levantar la infraestructura en Azure, configurar la seguridad para modelos de **Hugging Face** y desplegar un servicio de inferencia basado en **gRPC**.

## 1. Configuración de la Infraestructura Base

Antes de realizar el despliegue, es necesario recrear el entorno de ejecución en Azure. El clúster se configura con el complemento de monitoreo activado para habilitar las métricas en tiempo real.

```bash
# 1. Crear el Grupo de Recursos
az group create --name RG_Microproyecto2 --location centralus

# 2. Crear el clúster AKS con monitoreo habilitado
az aks create \
    --resource-group RG_Microproyecto2 \
    --name ClusterIA \
    --location centralus \
    --node-count 2 \
    --node-vm-size Standard_D2s_v3 \
    --enable-addons monitoring \
    --generate-ssh-keys

# 3. Vincular credenciales a la terminal
az aks get-credentials --resource-group RG_Microproyecto2 --name ClusterIA

```

---

## 2. Gestión de Secretos y Seguridad

Para que el contenedor pueda descargar el modelo `Ateeqq/ai-vs-human-image-detector` de forma eficiente y autenticada, se debe crear un secreto de Kubernetes con el token de Hugging Face.

```bash
# Crear secreto para autenticación en Hugging Face Hub
# Reemplazar 'tu_token_aqui' con un token con permisos de lectura (Read)
kubectl create secret generic hf-token --from-literal=HF_TOKEN='tu_token_aqui'

```

---

## 3. Despliegue del Servicio de Inferencia (IA vs Real)

El siguiente bloque genera el manifiesto de Kubernetes y lo aplica al clúster. Se utiliza un servicio tipo **LoadBalancer** para exponer el puerto gRPC (**50051**) a internet.

```bash
cat <<EOF > inferenceAIimage.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-ai-deployment
  labels:
    app: inference-ai
spec:
  replicas: 2
  selector:
    matchLabels:
      app: inference-ai
  template:
    metadata:
      labels:
        app: inference-ai
    spec:
      containers:
        - name: inference-ai-container
          image: davids117/aiimagerecognition:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 50051
              name: grpc
              protocol: TCP
          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: HF_TOKEN
            - name: HUGGINGFACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: HF_TOKEN
          resources:
            requests:
              cpu: "250m"
              memory: "512Mi"
            limits:
              cpu: "1000m"
              memory: "2Gi"
          readinessProbe:
            tcpSocket:
              port: 50051
            initialDelaySeconds: 15
            periodSeconds: 10
          livenessProbe:
            tcpSocket:
              port: 50051
            initialDelaySeconds: 30
            periodSeconds: 20
---
apiVersion: v1
kind: Service
metadata:
  name: inference-ai-service
  labels:
    app: inference-ai
spec:
  type: LoadBalancer
  selector:
    app: inference-ai
  ports:
    - name: grpc
      port: 50051
      targetPort: 50051
      protocol: TCP
EOF

# Aplicar los manifiestos
kubectl apply -f inferenceAIimage.yaml

```

---

## 4. Verificación y Monitoreo (Punto 4)

Una vez desplegada la aplicación, se debe validar que los Pods alcancen el estado `Ready` (esto puede tardar unos minutos debido a la descarga del modelo) y obtener la IP pública para la GUI.

### Comandos de Inspección

* **Estado de los Pods:** `kubectl get pods -w`
* **Obtener IP Pública (gRPC):** `kubectl get svc inference-ai-service`
* **Consumo de Recursos:** `kubectl top pods`

> **Nota Técnica:** El modelo utiliza aproximadamente **1.2 GB** de RAM en estado de reposo tras la carga inicial de los pesos. Se recomienda monitorear que el consumo no exceda el límite de **2Gi** durante las pruebas de inferencia.

---

# Punto 4: Monitoreo y Observabilidad del Clúster

Este apartado documenta la configuración necesaria para habilitar la recolección de métricas de rendimiento (CPU y Memoria) tanto de los nodos físicos como de los Pods que ejecutan el modelo de Deep Learning.

## 1. Registro de Proveedores de Recursos (Resource Providers)

Antes de habilitar el monitoreo en AKS, es obligatorio registrar los siguientes espacios de nombres en la suscripción de Azure. Sin estos registros, el clúster no puede comunicarse con los servicios de telemetría.

Ejecutar en la terminal de Azure:

```bash
# Habilita el motor de métricas de Azure Monitor
az provider register --namespace microsoft.insights

# Habilita el almacenamiento de logs y análisis operacional
az provider register --namespace Microsoft.OperationalInsights

```

> **Nota:** Se debe verificar que el estado sea `Registered` usando el comando:
> `az provider show -n microsoft.insights --query registrationState`

## 2. Despliegue del Clúster con Soporte de Monitoreo

Para habilitar el servidor de métricas de forma nativa, se debe incluir el addon de `monitoring` durante la creación del clúster. Esto instala automáticamente el agente de Log Analytics en los nodos.

```bash
az aks create \
    --resource-group RG_Microproyecto2 \
    --name ClusterIA \
    --location centralus \
    --node-count 2 \
    --node-vm-size Standard_D2s_v3 \
    --enable-addons monitoring \
    --generate-ssh-keys

```

## 3. Comandos de Verificación de Telemetría

Una vez desplegada la aplicación del Punto 2, se utilizan los siguientes comandos para monitorear el consumo de hardware. Estos datos son críticos para entender el impacto del modelo **ResNet20** en la infraestructura.

### A. Consumo de Nodos (Infraestructura)

Permite visualizar el porcentaje de carga de las máquinas virtuales `Standard_D2s_v3`.

```bash
kubectl top nodes

```

### B. Consumo de Pods (Aplicación de IA)

Permite observar cuántos milicores de CPU y cuántos Megabytes de RAM consume cada réplica del clasificador de imágenes en tiempo real.

```bash
kubectl top pods

```

## 4. Arquitectura de Observabilidad

La habilitación de estos registros permite que los datos fluyan desde el contenedor de Docker (en el nodo de AKS) hacia **Azure Monitor Container Insights**, permitiendo una supervisión detallada sin necesidad de instalar agentes adicionales manualmente dentro de la imagen.