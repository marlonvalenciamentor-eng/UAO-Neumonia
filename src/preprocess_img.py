"""
=============================================================================
MÓDULO 2: preprocess_img.py
ARQUITECTURA: Alta Cohesión y Bajo Acoplamiento (ArjanCodes / Clean Code)
RESPONSABILIDAD ÚNICA:
    Transformación morfológica, ecualización adaptativa de histograma y
    normalización numérica de matrices de imagen para la red convolucional.

OBJETIVO ESPECÍFICO DE LA FUNCIÓN:
preprocess_image(array: np.ndarray) -> np.ndarray:
    1. Valida el tipo y dimensiones del arreglo de entrada.
    2. Redimensiona espacialmente a 512x512 píxeles.
    3. Convierte a escala de grises de 1 solo canal (si viene en 3 canales).
    4. Aplica ecualización CLAHE (Contrast Limited Adaptive Histogram Equalization)
       para resaltar estructuras óseas y opacidades pulmonares.
    5. Normaliza los valores de intensidad al rango flotante [0.0, 1.0].
    6. Expande dimensiones a formato de batch (Tensor) de 4D: (1, 512, 512, 1).

ENTREGA:
    Tensor NumPy de 4 dimensiones (float32) con forma (1, 512, 512, 1).
=============================================================================
"""

import cv2
import numpy as np


def preprocess_image(array: np.ndarray) -> np.ndarray:
    """
    Preprocesa una imagen radiográfica (NumPy array) para la entrada de la CNN.
    
    Parámetros:
        array (np.ndarray): Matriz numérica de la imagen original.
        
    Retorna:
        np.ndarray: Tensor 4D de forma (1, 512, 512, 1) normalizado entre 0.0 y 1.0.
    """
    # 1. Validación de tipo de dato
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Se esperaba un numpy.ndarray, pero se recibió: {type(array)}")

    # 2. Validación de dimensiones
    if array.size == 0 or len(array.shape) < 2:
        raise ValueError("El arreglo de entrada está vacío o no tiene dimensiones de imagen válidas.")

    # 3. Redimensionamiento espacial estándar (512 x 512)
    resized = cv2.resize(array, (512, 512), interpolation=cv2.INTER_AREA)

    # 4. Conversión a escala de grises (1 canal)
    if len(resized.shape) == 3:
        if resized.shape[2] == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        elif resized.shape[2] == 1:
            gray = resized[:, :, 0]
        else:
            raise ValueError(f"Número de canales no soportado: {resized.shape[2]}")
    else:
        gray = resized

    # Asegurar que los datos para CLAHE sean enteros de 8 bits (0 - 255)
    if gray.dtype != np.uint8:
        max_val = gray.max()
        if max_val > 0:
            gray = np.uint8((gray / max_val) * 255.0)
        else:
            gray = np.zeros_like(gray, dtype=np.uint8)

    # 5. Ecualización adaptativa de contraste local (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    equalized = clahe.apply(gray)

    # 6. Normalización flotante en el rango [0.0, 1.0]
    normalized = equalized.astype(np.float32) / 255.0

    # 7. Expansión a 4 dimensiones para formato de batch de Keras/TensorFlow: (1, 512, 512, 1)
    batch_tensor = np.expand_dims(normalized, axis=-1)  # (512, 512, 1)
    batch_tensor = np.expand_dims(batch_tensor, axis=0)  # (1, 512, 512, 1)

    return batch_tensor
