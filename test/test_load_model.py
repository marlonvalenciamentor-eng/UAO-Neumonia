"""
Pruebas unitarias para el Módulo 3: load_model.py
"""
import pytest
import tensorflow as tf

from src.load_model import load_cnn_model, validate_model_integrity


def test_load_model_success():
    """Valida la carga exitosa y el tipo de objeto retornado."""
    model = load_cnn_model()
    assert isinstance(model, tf.keras.Model), "Debe ser una instancia de tf.keras.Model"


def test_model_input_and_output_dimensions():
    """Valida que las dimensiones de entrada y salida coincidan con el protocolo clínico."""
    model = load_cnn_model()
    
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    assert input_shape[1:] == (512, 512, 1), "La entrada debe ser (512, 512, 1)"

    output_shape = model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    assert output_shape[-1] == 3, "La salida debe tener 3 clases diagnósticas"


def test_model_has_gradcam_layer():
    """Valida que el modelo contenga la capa convolucional 'conv10_thisone'."""
    model = load_cnn_model()
    layer = model.get_layer("conv10_thisone")
    assert layer is not None, "La capa conv10_thisone debe existir en el modelo"


def test_load_model_file_not_found():
    """Valida que lance FileNotFoundError si se busca un modelo que no existe."""
    with pytest.raises(FileNotFoundError):
        load_cnn_model("modelo_fantasma_inexistente.h5")


def test_validate_model_integrity_wrong_layer():
    """Valida que la función de integridad detecte si falta una capa obligatoria."""
    model = load_cnn_model()
    with pytest.raises(ValueError, match="No se encontró la capa"):
        validate_model_integrity(model, required_layer="capa_que_no_existe_123")
