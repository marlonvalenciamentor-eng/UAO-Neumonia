"""Lectura de imagenes radiograficas.

Responsabilidad unica: convertir archivos DICOM o JPG en arreglos NumPy
listos para el resto del pipeline. No preprocesa ni predice.
"""

import cv2
import numpy as np
import pydicom
from PIL import Image


def _normalizar_a_uint8(arreglo):
    """Escala un arreglo a rango 0-255 en formato uint8."""
    imagen = arreglo.astype(float)
    maximo = imagen.max()
    if maximo == 0:
        return np.zeros_like(imagen, dtype=np.uint8)
    imagen = (np.maximum(imagen, 0) / maximo) * 255.0
    return np.uint8(imagen)


def read_dicom_file(path):
    """Lee un archivo DICOM.

    Args:
        path: ruta al archivo .dcm

    Returns:
        tuple: (arreglo RGB para el modelo, imagen PIL para la interfaz)
    """
    dataset = pydicom.dcmread(path)
    arreglo = dataset.pixel_array
    imagen_pil = Image.fromarray(arreglo)
    imagen = _normalizar_a_uint8(arreglo)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
    return imagen_rgb, imagen_pil


def read_jpg_file(path):
    """Lee un archivo JPG o PNG.

    Args:
        path: ruta al archivo de imagen

    Returns:
        tuple: (arreglo para el modelo, imagen PIL para la interfaz)

    Raises:
        FileNotFoundError: si OpenCV no logra leer el archivo
    """
    imagen = cv2.imread(path)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    arreglo = np.asarray(imagen)
    imagen_pil = Image.fromarray(arreglo)
    return _normalizar_a_uint8(arreglo), imagen_pil