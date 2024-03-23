# Proyecto de Métricas de Segmentación

Este proyecto calcula varias métricas para evaluar la precisión de segmentaciones de imágenes. Las métricas calculadas incluyen MIoU, SSIM, F1 y DICE.

## Instalación de Dependencias

Antes de ejecutar el script, asegúrate de instalar todas las dependencias necesarias. Puedes instalarlas ejecutando el siguiente comando en tu terminal:

```bash
pip install -r requirements.txt
```

## Ejecutar el archivo metrics.py

```bash
python metricas.py "ruta_mascara_verdadera.png" "ruta_mascara_generada.png"
```
