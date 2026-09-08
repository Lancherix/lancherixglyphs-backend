#!/usr/bin/env python3

"""
lancherix_scanner_preprocessor.py

Convierte una fotografía en una imagen tipo scanner:
- blancos más blancos
- negros más negros
- elimina gran parte de los grises
- conserva encuadre y geometría
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


_MIN_LONG_SIDE = 1600  # px - upscale interno para fotos chicas (ej.
                       # webcam 480x640), asi el resto del pipeline
                       # tiene mas pixeles por modulo para trabajar.


def scanner_effect(image):
    # ---------------------------------------------------------
    # 0. Upscale si la imagen entra chica (ej. webcam a 480x640).
    #    Fotos ya grandes (celular) pasan sin cambios.
    # ---------------------------------------------------------

    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side < _MIN_LONG_SIDE:
        scale = _MIN_LONG_SIDE / long_side
        image = cv2.resize(
            image,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )

    # ---------------------------------------------------------
    # 1. Convertir a escala de grises
    # ---------------------------------------------------------

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # 2. Suavizado que preserva bordes.
    # ---------------------------------------------------------

    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # ---------------------------------------------------------
    # 3. Aplanar la iluminacion (esto es lo que arregla las
    #    sombras).
    #
    #    Una sombra real hace que una zona del papel tenga un
    #    gris mas oscuro que el resto, aunque sea "blanco" de
    #    verdad. Un threshold GLOBAL (paso 5) no puede distinguir
    #    "blanco con sombra" de "negro real" si usa un solo corte
    #    para toda la imagen.
    #
    #    La solucion es estimar el "fondo" (la iluminacion de
    #    fondo, sin detalle) con un blur MUY grande, y dividir la
    #    imagen original por ese fondo. Esto normaliza el brillo:
    #    una zona con sombra y una zona sin sombra, ambas sobre
    #    papel blanco, terminan con valores similares despues de
    #    la division -- la sombra "desaparece" antes de llegar al
    #    threshold.
    # ---------------------------------------------------------

    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    # Evita division por cero en zonas completamente negras.
    background = np.where(background == 0, 1, background)

    normalized = (gray.astype(np.float32) / background.astype(np.float32)) * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------
    # 4. Ecualizacion de contraste LOCAL (CLAHE), sobre la imagen
    #    ya normalizada -- refuerza el contraste real (texto,
    #    modulos del codigo) sin reintroducir el gradiente de
    #    sombra que acabamos de aplanar.
    # ---------------------------------------------------------

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    normalized = clahe.apply(normalized)

    # ---------------------------------------------------------
    # 5. Aumentar contraste
    #
    #    Empuja los valores claros hacia 255
    #    y los oscuros hacia 0.
    # ---------------------------------------------------------

    low = np.percentile(normalized, 2)
    high = np.percentile(normalized, 98)

    if high > low:
        contrast = np.clip(
            (normalized.astype(np.float32) - low)
            * 255.0
            / (high - low),
            0,
            255
        ).astype(np.uint8)
    else:
        contrast = normalized

    # ---------------------------------------------------------
    # 6. Threshold OTSU
    #
    #    Convierte finalmente la imagen en:
    #
    #       0   = negro
    #       255 = blanco
    #
    #    El umbral se calcula automáticamente. Como ya llega con
    #    la iluminacion aplanada, un solo corte global ahora si
    #    es representativo de toda la imagen.
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
    print(f"Input size  : {image.shape[1]}x{image.shape[0]}")
    print(f"Output size : {result.shape[1]}x{result.shape[0]}")
    print()
    print("Blancos -> blanco")
    print("Negros  -> negro")
    print("Encuadre conservado")
    print("Geometria conservada")
    print()


if __name__ == "__main__":
    main()