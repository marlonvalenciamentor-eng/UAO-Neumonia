"""
Pruebas unitarias para el Módulo 4: grad_cam.py
"""
import pytest
import numpy as np

from src.grad_cam import predict_and_explain, generate_gradcam_heatmap
from src.load_model import load_cnn_model
from src.preprocess_img import preprocess_image
from src.read_img import read_dicom_file, read_jpg_file

SAMPLE_DICOM = "data/DICOM/normal (2).dcm"
SAMPLE_JPG = "data/JPG/normal/NORMAL2-IM-1145-0001.jpeg"


@pytest.fixture(scope="module")
def loaded_model():
    """Fixture que carga el modelo una sola vez para todas las pruebas del módulo."""
    return load_cnn_model()


def test_predict_and_explain_dicom(loaded_model):
    """Valida la inferencia y Grad-CAM completo sobre un archivo DICOM."""
    array, _ = read_dicom_file(SAMPLE_DICOM)
    label, proba, heatmap = predict_and_explain(array, model=loaded_model)

    assert label in ["bacteriana", "normal", "viral"], f"Clase desconocida: {label}"
    assert 0.0 <= proba <= 100.0, f"Probabilidad fuera de rango: {proba}"
    assert isinstance(heatmap, np.ndarray), "El heatmap debe ser un numpy array"
    assert heatmap.shape == (512, 512, 3), f"Forma esperada (512, 512, 3), recibida {heatmap.shape}"
    assert heatmap.dtype == np.uint8, "El tipo de dato del heatmap debe ser uint8"


def test_predict_and_explain_jpg(loaded_model):
    """Valida la inferencia y Grad-CAM sobre una imagen JPG."""
    array, _ = read_jpg_file(SAMPLE_JPG)
    label, proba, heatmap = predict_and_explain(array, model=loaded_model)

    assert label in ["bacteriana", "normal", "viral"]
    assert 0.0 <= proba <= 100.0
    assert heatmap.shape == (512, 512, 3)


def test_generate_gradcam_heatmap_dimensions(loaded_model):
    """Valida que el mapa de calor crudo tenga dimensiones 512x512 y esté acotado entre 0 y 1."""
    array, _ = read_dicom_file(SAMPLE_DICOM)
    batch = preprocess_image(array)
    heatmap_raw = generate_gradcam_heatmap(loaded_model, batch)

    assert heatmap_raw.shape == (512, 512)
    assert heatmap_raw.min() >= 0.0
    assert heatmap_raw.max() <= 1.0


def test_predict_invalid_layer(loaded_model):
    """Valida que capture error si se solicita una capa convolucional inexistente."""
    array, _ = read_dicom_file(SAMPLE_DICOM)
    with pytest.raises(ValueError):
        predict_and_explain(array, model=loaded_model, layer_name="capa_inexistente")
