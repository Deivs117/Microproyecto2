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
curl -X POST -F "img=@assets/horse.jpg" http://<IP_AZURE>/predict

```

---

### 4. Notas Técnicas para Sustentación

* **Corrección de Imagen:** Se utiliza el repositorio `davids117/image-classifier:v1`.
* **Ajuste de Código:** Se configuró `pretrained=True` para que el contenedor descargue los pesos del modelo automáticamente al iniciar.
* **Infraestructura:** El clúster corre en 2 nodos `Standard_D2s_v3` en la región `centralus`.

