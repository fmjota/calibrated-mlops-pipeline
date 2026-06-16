# Podman vs Docker — por qué y cómo se ejecuta aquí

Este proyecto se empaqueta con un `Dockerfile` y un `docker-compose.yml` **estándar OCI**,
y se **ejecuta con Podman**. Este documento explica la diferencia entre ambos, por qué se
eligió Podman en este entorno y cómo correr el contenedor con uno u otro.

## Qué son

- **Docker.** La herramienta que popularizó los contenedores. Usa un **daemon** central
  (`dockerd`) que corre como **root** y atiende todas las peticiones del cliente `docker`.
- **Podman.** Motor de contenedores compatible con OCI, **sin daemon** y **rootless** por
  defecto. La CLI es casi idéntica a la de Docker (`alias docker=podman` suele bastar) y
  lee los mismos `Dockerfile`/`docker-compose.yml`.

## Diferencias clave

| Aspecto | Docker | Podman |
|---|---|---|
| Arquitectura | Daemon central (`dockerd`) | **Sin daemon**: cada contenedor es un proceso hijo |
| Privilegios | El daemon corre como root | **Rootless** por defecto (más seguro) |
| Seguridad | Superficie de ataque en el daemon root | Sin daemon root; aísla por usuario |
| Compose | `docker compose` (plugin v2 incluido) | `podman-compose` o `podman compose` (proveedor externo) |
| Pods | No nativo | **Pods** nativos (grupos de contenedores, estilo Kubernetes) |
| Imágenes/Registros | OCI | OCI (mismas imágenes) |
| Integración Fedora/RHEL | Repo externo | **Incluido y soportado** por Red Hat/Fedora |

La idea central: **el formato es el mismo** (imágenes OCI, `Dockerfile`, `compose`), así
que el `docker-compose.yml` de este repo funciona en ambos sin cambios de fondo.

## Por qué Podman en este entorno (y no Docker)

1. **Fedora 44 ya trae Podman** (5.8.2) instalado; Docker no estaba.
2. **Sin sudo ni daemon root.** Podman corre rootless: no hay que instalar ni habilitar un
   servicio privilegiado. Encaja con la regla del proyecto de **no tocar el sistema** salvo
   con OK explícito.
3. **El repo oficial de Docker CE suele tardar** en soportar una Fedora recién salida, lo
   que complica instalar Docker en Fedora 44.
4. **Mismo artefacto.** Como todo es OCI estándar, quien prefiera Docker corre el mismo
   `docker compose up` sin tocar nada (ver más abajo).

> En resumen: se eligió Podman por disponibilidad, seguridad (rootless) y cero fricción de
> instalación, sin sacrificar portabilidad — el empaquetado sigue siendo Docker-compatible.

## Cómo se ejecuta

### Con Podman (lo usado aquí)

```bash
# 1. Proveedor de compose a nivel usuario (sin sudo)
uv tool install podman-compose

# 2. Construir y levantar solo la API (rootless)
podman-compose build api
podman-compose up -d api

# 3. Probar
curl localhost:8000/health
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{...}'

# 4. Bajar
podman-compose down
```

### Con Docker (si está instalado)

```bash
docker compose up --build        # mismo archivo, mismo resultado
```

## Detalles del entorno que hubo que resolver

- **SELinux (Fedora Enforcing).** Un volumen montado (`./artifacts`) no es legible por el
  contenedor sin **relabel**. Se añade la etiqueta `z` en el `docker-compose.yml`
  (`./artifacts:/app/artifacts:ro,z`). **Docker ignora `z` sin problema**, así que el
  archivo sigue siendo portable.
- **`libgomp1` para LightGBM.** La imagen base `python:3.12-slim` no incluye el runtime de
  OpenMP que LightGBM carga al importar. El `Dockerfile` instala `libgomp1` con apt.
- **Rootless + puertos.** Se mapean puertos > 1024 (8000, 5000), que Podman rootless
  expone sin privilegios.

## Glosario rápido

- **OCI** (Open Container Initiative): estándar de imágenes y runtime que hace
  intercambiables a Docker y Podman.
- **Rootless**: contenedores que corren bajo tu usuario, sin privilegios de root.
- **Daemon**: proceso de fondo siempre activo (Docker lo tiene; Podman no).
- **Relabel SELinux (`z`/`Z`)**: ajusta etiquetas de seguridad de un volumen para que el
  contenedor pueda leerlo (`z` compartido, `Z` privado).
