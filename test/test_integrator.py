"""
Pruebas unitarias para el Módulo 5: integrator.py
"""
import pytest
import numpy as np
from PIL import Image

from src.integrator import process_and_diagnose, predict, get_or_load_model
from src.read_img import read_dicom_file

SAMPLE_DICOM = "data/DICOM/normal (2).dcm"
SAMPLE_JPG = "data/JPG/normal/NORMAL2-IM-1145-0001.jpeg"


def test_process_and_diagnose_dicom_path():
    """Valida la orquestación completa recibiendo una ruta de archivo DICOM."""
    results = process_and_diagnose(SAMPLE_DICOM)

    assert isinstance(results, dict), "Debe retornar un diccionario"
    assert results["label"] in ["bacteriana", "normal", "viral"]
    assert 0.0 <= results["probability"] <= 100.0
    assert isinstance(results["heatmap"], np.ndarray)
    assert isinstance(results["display_image"], Image.Image)


def test_process_and_diagnose_jpg_path():
    """Valida la orquestación completa recibiendo una ruta de archivo JPG."""
    results = process_and_diagnose(SAMPLE_JPG)

    assert results["label"] in ["bacteriana", "normal", "viral"]
    assert 0.0 <= results["probability"] <= 100.0
    assert results["heatmap"].shape == (512, 512, 3)


def test_process_and_diagnose_numpy_array():
    """Valida la orquestación recibiendo una matriz NumPy ya en memoria."""
    array, _ = read_dicom_file(SAMPLE_DICOM)
    results = process_and_diagnose(array)

    assert results["label"] in ["bacteriana", "normal", "viral"]
    assert isinstance(results["heatmap"], np.ndarray)


def test_predict_bridge_function():
    """Valida la función puente predict(array) que usa la interfaz gráfica."""
    array, _ = read_dicom_file(SAMPLE_DICOM)
    label, proba, heatmap = predict(array)

    assert label in ["bacteriana", "normal", "viral"]
    assert isinstance(proba, float)
    assert isinstance(heatmap, np.ndarray)


def test_invalid_input_type():
    """Valida que capture entradas que no sean ni str ni ndarray."""
    with pytest.raises(TypeError):
        process_and_diagnose(12345)


def test_model_caching():
    """Valida que el modelo se mantenga en memoria y retorne la misma instancia."""
    model_1 = get_or_load_model()
    model_2 = get_or_load_model()
    assert model_1 is model_2, "Debe ser la misma instancia en caché"
