"""Carga del modelo de deteccion de neumonia.

Responsabilidad unica: entregar el modelo entrenado. La cache garantiza
que el archivo .h5 se lea del disco una sola vez por ejecucion, sin
importar cuantos modulos lo pidan.
"""

import os
from functools import lru_cache

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf

RUTA_MODELO = "conv_MLP_84.h5"


@lru_cache(maxsize=1)
def load_model(ruta=RUTA_MODELO):
    """Carga el modelo entrenado desde disco.

    Args:
        ruta: ubicacion del archivo .h5

    Returns:
        El modelo Keras listo para inferencia.

    Raises:
        FileNotFoundError: si el archivo no existe en la ruta indicada.
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontro el modelo en '{ruta}'. "
            "Descarguelo del campus y ubiquelo en la raiz del proyecto."
        )
    return tf.keras.models.load_model(ruta)