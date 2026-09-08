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

_MIN_LONG_SIDE = 1600  # px - la deteccion de corner markers, Hough y la
                       # lectura de la grilla dependen de tener suficientes
                       # pixeles por modulo. A 480x640, cada modulo del
                       # codigo ocupa apenas un puñado de pixeles reales;
                       # este upscale interno (via interpolacion cubica)
                       # le da a todo el resto del pipeline mas pixeles
                       # para trabajar, sin depender de que el navegador
                       # entregue una foto mas grande.


def scanner_effect(image):
    # ---------------------------------------------------------
    # 0. Upscale si la imagen entra chica (ej. webcam a 480x640).
    #    No inventa detalle real, pero evita que blur/contraste/
    #    threshold operen sobre muy pocos pixeles por modulo.
    #    Fotos ya grandes (celular, 3024x4032) pasan sin cambios.
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
    #
    #    Con pocos pixeles por modulo, un blur gaussiano comun
    #    difumina los bordes de los markers casi tanto como el
    #    ruido que busca eliminar. El filtro bilateral suaviza
    #    ruido de sensor/compresion manteniendo los bordes nitidos
    #    -- justo lo que Hough necesita despues para ubicar los
    #    lados de los markers con precision.
    # ---------------------------------------------------------
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # ---------------------------------------------------------
    # 3. Ecualizacion de contraste LOCAL (CLAHE).
    #
    #    Las camaras web suelen tener auto-exposicion menos
    #    uniforme que un telefono (una esquina del cuadro puede
    #    quedar mas oscura). CLAHE normaliza el contraste por
    #    regiones para que el paso 4 (contraste global + Otsu) no
    #    se vea arrastrado por una zona mas oscura que el resto.
    # ---------------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ---------------------------------------------------------
    # 4. Aumentar contraste (igual que antes)
    # ---------------------------------------------------------
    low = np.percentile(gray, 2)
    high = np.percentile(gray, 98)

    if high > low:
        contrast = np.clip(
            (gray.astype(np.float32) - low) * 255.0 / (high - low),
            0, 255,
        ).astype(np.uint8)
    else:
        contrast = gray

    # ---------------------------------------------------------
    # 5. Threshold OTSU (igual que antes)
    # ---------------------------------------------------------
    _, result = cv2.threshold(
        contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
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