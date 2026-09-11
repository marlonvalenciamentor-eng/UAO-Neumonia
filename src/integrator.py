"""
=============================================================================
MÓDULO 5: integrator.py
ARQUITECTURA: Patrón Fachada / Orquestador (ArjanCodes / Clean Code)
RESPONSABILIDAD ÚNICA:
    Coordinar la ejecución secuencial de los módulos (lectura, preprocesamiento,
    modelo e inferencia Grad-CAM) y unificar las salidas tanto para la interfaz
    gráfica (GUI Tkinter) como para la línea de comandos (CLI).

FLUJO DEL PIPELINE:
    Entrada (Ruta o Matriz)
        ⬇️
    read_img.py        -> Extrae la matriz y la imagen PIL
        ⬇️
    load_model.py      -> Carga el modelo con validación (en caché)
        ⬇️
    grad_cam.py        -> Preprocesa, infiere probabilidades y calcula Heatmap
        ⬇️
    Salida Unificada   -> Diccionario estructurado + tupla compatible
=============================================================================
"""

import sys
import argparse
import numpy as np
from PIL import Image

from src.read_img import read_file
from src.load_model import load_cnn_model
from src.grad_cam import predict_and_explain

# Variable global a nivel de módulo para mantener el modelo en memoria (Singleton / Caché)
_CACHED_MODEL = None


def get_or_load_model():
    """
    Retorna el modelo en memoria. Si aún no está cargado, lo carga y lo almacena en caché.
    Evita recargar 113 MB de disco en cada predicción.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        _CACHED_MODEL = load_cnn_model()
    return _CACHED_MODEL


def process_and_diagnose(input_data, model=None) -> dict:
    """
    Función principal de integración:
    
    Acepta:
        - input_data: Ruta a un archivo (.dcm, .jpg, .png) O un arreglo NumPy en memoria.
        - model: Modelo pre-cargado (opcional).
        
    Retorna un diccionario clínico unificado:
        {
            "label": str,              # 'bacteriana', 'normal', 'viral'
            "probability": float,      # Porcentaje (0.0 a 100.0)
            "heatmap": np.ndarray,     # Imagen con Grad-CAM superpuesto (512x512x3, uint8)
            "original_array": np.ndarray, # Matriz original
            "display_image": Image.Image  # Objeto PIL para mostrar en Tkinter/Streamlit
        }
    """
    # 1. Resolver el modelo (usar el suministrado o el de caché)
    active_model = model if model is not None else get_or_load_model()

    # 2. Paso de lectura según el tipo de entrada recibida
    if isinstance(input_data, str):
        original_array, display_image = read_file(input_data)
    elif isinstance(input_data, np.ndarray):
        original_array = input_data
        # Si no vino el objeto PIL, lo construimos a partir de la matriz
        if len(original_array.shape) == 2:
            display_image = Image.fromarray(original_array)
        else:
            display_image = Image.fromarray(original_array.astype(np.uint8))
    else:
        raise TypeError(
            f"Tipo de entrada no soportado: {type(input_data)}. "
            "Se esperaba una ruta (str) o una matriz NumPy (np.ndarray)."
        )

    # 3. Ejecutar predicción y explicabilidad Grad-CAM
    label, proba, heatmap = predict_and_explain(original_array, model=active_model)

    # 4. Retornar contrato de datos unificado
    return {
        "label": label,
        "probability": proba,
        "heatmap": heatmap,
        "original_array": original_array,
        "display_image": display_image
    }


# Función compatible con la firma que usaba el detector_neumonia.py original
def predict(array: np.ndarray):
    """
    Función puente compatible con la interfaz gráfica original.
    Retorna la tupla clásica: (label, proba, heatmap)
    """
    results = process_and_diagnose(array)
    return results["label"], results["probability"], results["heatmap"]


def main_cli():
    """
    Punto de entrada para ejecutar el sistema por consola (CLI),
    cumpliendo con el requisito de la guía: 'salidas para interfaz gráfica / CLI'.
    """
    parser = argparse.ArgumentParser(
        description="Sistema de Detección de Neumonía y Explicabilidad (CLI)"
    )
    parser.add_argument("image_path", help="Ruta de la imagen médica (.dcm, .jpg, .png)")
    args = parser.parse_args()

    print(f"🔬 Analizando radiografía: {args.image_path}...")
    try:
        results = process_and_diagnose(args.image_path)
        print("\n" + "=" * 45)
        print("  RESULTADOS DEL DIAGNÓSTICO CLÍNICO (IA)")
        print("=" * 45)
        print(f"  Diagnóstico : {results['label'].upper()}")
        print(f"  Confianza   : {results['probability']:.2f}%")
        print(f"  Heatmap     : Generado con éxito ({results['heatmap'].shape})")
        print("=" * 45 + "\n")
    except Exception as e:
        print(f"❌ Error al procesar la imagen: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
