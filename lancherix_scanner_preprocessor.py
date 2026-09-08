#!/usr/bin/env python3

"""
lancherix_scanner_preprocessor.py

Convierte una fotografía en una imagen tipo scanner:
- blancos más blancos
- negros más negros
- elimina gran parte de los grises
- conserva tamaño, encuadre y geometría
- NO recorta
- NO rectifica
- NO detecta markers

Uso:
    python3 lancherix_scanner_preprocessor.py foto.png

Salida:
    foto_scanner.png
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def scanner_effect(image):
    # ---------------------------------------------------------
    # 1. Convertir a escala de grises
    # ---------------------------------------------------------

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # 2. Suavizado MUY ligero
    #    Elimina pequeñas imperfecciones del papel.
    # ---------------------------------------------------------

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # ---------------------------------------------------------
    # 3. Aumentar contraste
    #
    #    Empuja los valores claros hacia 255
    #    y los oscuros hacia 0.
    # ---------------------------------------------------------

    low = np.percentile(gray, 2)
    high = np.percentile(gray, 98)

    if high > low:
        contrast = np.clip(
            (gray.astype(np.float32) - low)
            * 255.0
            / (high - low),
            0,
            255
        ).astype(np.uint8)
    else:
        contrast = gray

    # ---------------------------------------------------------
    # 4. Threshold OTSU
    #
    #    Convierte finalmente la imagen en:
    #
    #       0   = negro
    #       255 = blanco
    #
    #    El umbral se calcula automáticamente.
    # ---------------------------------------------------------

    _, result = cv2.threshold(
        contrast,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return result


def main():

    if len(sys.argv) != 2:
        print("Uso:")
        print("  python3 lancherix_scanner_preprocessor.py foto.png")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"Error: no existe el archivo: {input_path}")
        sys.exit(1)

    image = cv2.imread(
        str(input_path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        print(f"Error: no se pudo leer: {input_path}")
        sys.exit(1)

    result = scanner_effect(image)

    output_path = input_path.with_name(
        f"{input_path.stem}_scanner.png"
    )

    if not cv2.imwrite(
        str(output_path),
        result
    ):
        print(f"Error: no se pudo guardar: {output_path}")
        sys.exit(1)

    print()
    print("Lancherix Scanner Preprocessor")
    print("==============================")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Size  : {image.shape[1]}x{image.shape[0]}")
    print()
    print("Blancos -> blanco")
    print("Negros  -> negro")
    print("Encuadre conservado")
    print("Geometria conservada")
    print()


if __name__ == "__main__":
    main()