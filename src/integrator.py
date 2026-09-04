"""Coordinacion del pipeline de deteccion de neumonia.

Responsabilidad unica: orquestar lectura, preprocesamiento, inferencia y
Grad-CAM, y unificar las salidas en una sola respuesta. Es el unico punto
que la interfaz grafica necesita conocer.
"""

import numpy as np

from src.grad_cam import grad_cam
from src.load_model import load_model
from src.preprocess_img import preprocess

ETIQUETAS = {0: "bacteriana", 1: "normal", 2: "viral"}


def predict(arreglo):
    """Ejecuta el pipeline completo sobre una radiografia.

    Args:
        arreglo: imagen RGB como arreglo NumPy

    Returns:
        tuple: (etiqueta, probabilidad en porcentaje, imagen con heatmap)
    """
    modelo = load_model()
    lote = preprocess(arreglo)

    predicciones = modelo.predict(lote, verbose=0)
    indice = int(np.argmax(predicciones))
    etiqueta = ETIQUETAS.get(indice, "desconocido")
    probabilidad = float(np.max(predicciones)) * 100

    mapa_calor = grad_cam(arreglo, modelo=modelo)
    return etiqueta, probabilidad, mapa_calor