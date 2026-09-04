"""Generacion del mapa de calor Grad-CAM.

Responsabilidad unica: producir la radiografia con el mapa de calor
superpuesto, senalando las regiones que mas pesaron en la prediccion.

Usa tf.GradientTape (modo eager) en lugar del modo grafo de TensorFlow 1,
lo que elimina la dependencia de tf.compat.v1 y sus advertencias.
"""

import cv2
import numpy as np
import tensorflow as tf

from src.load_model import load_model
from src.preprocess_img import preprocess

CAPA_CONVOLUCIONAL_FINAL = "conv10_thisone"
TAMANO_SALIDA = (512, 512)
TRANSPARENCIA = 0.8


def _calcular_mapa(modelo, lote, nombre_capa):
    """Calcula el mapa de activacion normalizado entre 0 y 1."""
    capa = modelo.get_layer(nombre_capa)
    modelo_grad = tf.keras.models.Model(modelo.inputs, [capa.output, modelo.output])

    with tf.GradientTape() as cinta:
        activaciones, predicciones = modelo_grad(lote)
        clase = tf.argmax(predicciones[0])
        salida = predicciones[:, clase]

    gradientes = cinta.gradient(salida, activaciones)
    pesos = tf.reduce_mean(gradientes, axis=(0, 1, 2))
    mapa = tf.reduce_sum(activaciones[0] * pesos, axis=-1)
    mapa = tf.maximum(mapa, 0)

    maximo = tf.reduce_max(mapa)
    if maximo > 0:
        mapa = mapa / maximo
    return mapa.numpy()


def grad_cam(arreglo, modelo=None, nombre_capa=CAPA_CONVOLUCIONAL_FINAL):
    """Superpone el mapa de calor sobre la radiografia original.

    Args:
        arreglo: imagen RGB como arreglo NumPy
        modelo: modelo ya cargado. Si es None se obtiene de load_model()
        nombre_capa: capa convolucional de la que se extraen las activaciones

    Returns:
        np.ndarray: imagen RGB con el heatmap superpuesto
    """
    modelo = modelo if modelo is not None else load_model()
    lote = preprocess(arreglo)

    mapa = _calcular_mapa(modelo, lote, nombre_capa)
    mapa = cv2.resize(mapa, TAMANO_SALIDA)
    mapa = np.uint8(255 * mapa)
    mapa = cv2.applyColorMap(mapa, cv2.COLORMAP_JET)

    imagen = cv2.resize(arreglo, TAMANO_SALIDA)
    superpuesta = cv2.add((mapa * TRANSPARENCIA).astype(np.uint8), imagen)
    return superpuesta.astype(np.uint8)[:, :, ::-1]