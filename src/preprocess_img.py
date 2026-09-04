"""Preprocesamiento de imagenes radiograficas.

Responsabilidad unica: dejar una imagen lista para el modelo.
Redimensiona a 512x512, convierte a escala de grises, aplica
ecualizacion CLAHE, normaliza y agrega las dimensiones de canal y lote.
"""

import cv2
import numpy as np

TAMANO_ENTRADA = (512, 512)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (4, 4)


def preprocess(arreglo):
    """Prepara una imagen para la entrada del modelo.

    Args:
        arreglo: imagen RGB como arreglo NumPy

    Returns:
        np.ndarray: lote de forma (1, 512, 512, 1) normalizado a 0-1
    """
    imagen = cv2.resize(arreglo, TAMANO_ENTRADA)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    imagen = clahe.apply(imagen)
    imagen = imagen / 255.0
    imagen = np.expand_dims(imagen, axis=-1)
    imagen = np.expand_dims(imagen, axis=0)
    return imagen