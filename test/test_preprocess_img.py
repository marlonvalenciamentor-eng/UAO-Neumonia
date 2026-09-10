"""
Pruebas unitarias para el Módulo 2: preprocess_img.py
"""
import pytest
import numpy as np

from src.preprocess_img import preprocess_image
from src.read_img import read_dicom_file

SAMPLE_DICOM = "data/DICOM/normal (2).dcm"


def test_preprocess_output_shape():
    """Valida que la salida tenga exactamente la forma (1, 512, 512, 1)."""
    array, _ = read_dicom_file(SAMPLE_DICOM)
    tensor = preprocess_image(array)

    assert isinstance(tensor, np.ndarray), "Debe retornar un numpy array"
    assert tensor.shape == (1, 512, 512, 1), f"Forma esperada (1, 512, 512, 1), recibida {tensor.shape}"


def test_preprocess_value_range():
    """Valida que los valores estén normalizados estrictamente entre 0.0 y 1.0."""
    array, _ = read_dicom_file(SAMPLE_DICOM)
    tensor = preprocess_image(array)

    assert tensor.dtype == np.float32, "El tipo de dato debe ser float32"
    assert tensor.min() >= 0.0, "El valor mínimo no puede ser menor a 0.0"
    assert tensor.max() <= 1.0, "El valor máximo no puede ser mayor a 1.0"


def test_preprocess_grayscale_input():
    """Valida que funcione correctamente si la imagen ya viene en escala de grises (2D)."""
    dummy_2d = np.ones((600, 600), dtype=np.uint8) * 128
    tensor = preprocess_image(dummy_2d)

    assert tensor.shape == (1, 512, 512, 1)


def test_preprocess_invalid_input_type():
    """Valida que lance TypeError si se pasa un string o lista en vez de ndarray."""
    with pytest.raises(TypeError):
        preprocess_image("esto_no_es_una_matriz")


def test_preprocess_empty_array():
    """Valida que lance ValueError si se pasa un arreglo vacío."""
    with pytest.raises(ValueError):
        preprocess_image(np.array([]))
