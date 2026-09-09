#!/usr/bin/env python3

"""
lancherix_scanner_preprocessor.py (v3)

Convierte una fotografia en una imagen tipo scanner:
- blancos mas blancos
- negros mas negros
- elimina gran parte de los grises
- conserva tamano, encuadre y geometria
- NO recorta
- NO rectifica
- NO detecta markers
- Robusto a iluminacion despareja (sombras, luz de lado)
- Rapido (opera a baja resolucion para estimar iluminacion)

Uso:
    python3 lancherix_scanner_preprocessor.py foto.png
Salida:
    foto_scanner.png
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def estimate_background(gray, close_frac=0.35, down_max=400):
    """
    Estima la iluminacion de fondo (sin las formas oscuras) usando
    un CLOSING morfologico (dilatar + erosionar) sobre una version
    reducida de la imagen.

    Por que closing y no un blur simple:
    Un blur promedia pixeles negros y blancos juntos, asi que dentro
    de una forma negra GRANDE, el "fondo" estimado tambien baja, y al
    dividir gray/fondo aparecen manchas blancas falsas dentro de las
    formas. El closing, en cambio, "rellena" los huecos oscuros mas
    chicos que el kernel con el valor claro que los rodea, sin
    promediar, y por eso no se contamina con las formas del codigo.

    Se trabaja en una version reducida (down_max) porque:
      1. Es mucho mas rapido.
      2. El kernel de closing necesita ser mas grande que las formas
         del codigo, lo cual es barato en baja resolucion y costoso
         en la resolucion original.
    """
    h, w = gray.shape
    scale = down_max / max(h, w)
    small = cv2.resize(
        gray,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )

    dh, dw = small.shape
    k = max(15, int(min(dh, dw) * close_frac) | 1)  # impar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    bg_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    bg_small = cv2.GaussianBlur(bg_small, (0, 0), sigmaX=k / 4)

    background = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)
    return background


def scanner_effect(image):
    # ---------------------------------------------------------
    # 1. Escala de grises + suavizado muy ligero
    # ---------------------------------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # ---------------------------------------------------------
    # 2. CORRECCION DE ILUMINACION
    #    Aplana sombras / luz de lado dividiendo por el fondo
    #    estimado. Esto es lo que arregla la inconsistencia con
    #    distintos tipos de luz.
    # ---------------------------------------------------------
    background = estimate_background(gray)
    background = np.maximum(background, 1).astype(np.float32)

    normalized = (gray.astype(np.float32) / background) * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------
    # 3. Estiramiento de contraste (percentiles 2-98)
    # ---------------------------------------------------------
    low = np.percentile(normalized, 2)
    high = np.percentile(normalized, 98)

    if high > low:
        contrast = np.clip(
            (normalized.astype(np.float32) - low) * 255.0 / (high - low),
            0,
            255,
        ).astype(np.uint8)
    else:
        contrast = normalized

    # ---------------------------------------------------------
    # 4. Threshold OTSU -> blanco y negro puros
    # ---------------------------------------------------------
    _, result = cv2.threshold(
        contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ---------------------------------------------------------
    # 5. Limpieza de ruido tipo sal y pimienta (kernel chico,
    #    no deforma los bordes de las formas)
    # ---------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

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

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: no se pudo leer: {input_path}")
        sys.exit(1)

    result = scanner_effect(image)

    output_path = input_path.with_name(f"{input_path.stem}_scanner.png")
    if not cv2.imwrite(str(output_path), result):
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
    print("Iluminacion corregida (closing morfologico + division)")
    print("Encuadre conservado")
    print("Geometria conservada")
    print()


if __name__ == "__main__":
    main()