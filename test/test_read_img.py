"""
Pruebas unitarias para el Módulo 1: read_img.py
"""
import pytest
import numpy as np
from PIL import Image

from src.read_img import read_dicom_file, read_jpg_file, read_file

# Rutas a imágenes de muestra que sabemos que existen
SAMPLE_DICOM = "data/DICOM/normal (2).dcm"
SAMPLE_JPG = "data/JPG/normal/NORMAL2-IM-1145-0001.jpeg"


def test_read_dicom_valid():
    """Valida la lectura correcta de un archivo DICOM existente."""
    array, img_pil = read_dicom_file(SAMPLE_DICOM)

    assert isinstance(array, np.ndarray), "Debe retornar un arreglo NumPy"
    assert isinstance(img_pil, Image.Image), "Debe retornar un objeto PIL Image"
    assert array.dtype == np.uint8, "El arreglo debe ser de tipo uint8"
    assert len(array.shape) == 3 and array.shape[2] == 3, "Debe tener 3 canales (RGB)"
    assert array.min() >= 0 and array.max() <= 255, "Los valores deben estar normalizados entre 0 y 255"


def test_read_dicom_not_found():
    """Valida que lance FileNotFoundError si el archivo DICOM no existe."""
    with pytest.raises(FileNotFoundError):
        read_dicom_file("ruta/inexistente/muestra.dcm")


def test_read_jpg_valid():
    """Valida la lectura correcta de una imagen JPG/JPEG existente."""
    array, img_pil = read_jpg_file(SAMPLE_JPG)

    assert isinstance(array, np.ndarray), "Debe retornar un arreglo NumPy"
    assert isinstance(img_pil, Image.Image), "Debe retornar un objeto PIL Image"
    assert array.dtype == np.uint8, "El arreglo debe ser de tipo uint8"
    assert len(array.shape) == 3 and array.shape[2] == 3, "Debe tener 3 canales (RGB)"


def test_read_jpg_not_found():
    """Valida que lance FileNotFoundError si la imagen JPG no existe."""
    with pytest.raises(FileNotFoundError):
        read_jpg_file("ruta/inexistente/muestra.jpeg")


def test_read_file_auto_detect_dicom():
    """Valida que la función fachada reconozca un archivo .dcm."""
    array, _ = read_file(SAMPLE_DICOM)
    assert isinstance(array, np.ndarray)


def test_read_file_auto_detect_jpg():
    """Valida que la función fachada reconozca un archivo .jpeg."""
    array, _ = read_file(SAMPLE_JPG)
    assert isinstance(array, np.ndarray)
