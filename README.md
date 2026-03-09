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