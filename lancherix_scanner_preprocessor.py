#!/usr/bin/env python3

"""
lancherix_scanner_preprocessor.py (v5)

Convierte una fotografia en una imagen tipo scanner:
- blancos mas blancos
- negros mas negros
- elimina gran parte de los grises
- conserva encuadre y geometria
- NO recorta
- NO rectifica
- NO detecta markers
- Robusto a iluminacion despareja (sombras, luz de lado)
- Rapido
- Nitido incluso en resoluciones chicas (ver nota de tamano abajo)

Uso:
    python3 lancherix_scanner_preprocessor.py foto.png
Salida:
    foto_scanner.png

---------------------------------------------------------------------
POR QUE HAY DOS CAMINOS (imagen chica vs imagen grande)
---------------------------------------------------------------------
La correccion de iluminacion (division por un fondo estimado con
closing morfologico) esta pensada para fotos de una hoja de
documento completa, donde el codigo ocupa una fraccion chica del
cuadro y sobra papel blanco alrededor para estimar bien el fondo.

En fotos chicas / de cerca (celular en baja resolucion, el codigo
ocupando casi todo el cuadro) esa correccion hace mas dano que bien:
el kernel de closing necesita ser enorme en relacion a la imagen
para "saltar por encima" de las formas negras, y a ese tamano
empieza a degradar asimetricamente detalles finos como el punto
blanco central de los corner markers (se probo empiricamente: el
mismo punto que mide ~12x12px en la foto original puede terminar en
9x9px de un lado y 3x3px del otro despues de la correccion de
iluminacion completa). Por eso para imagenes chicas se usa un camino
mas simple y conservador: sin correccion de fondo, sin limpieza
morfologica final agresiva.

---------------------------------------------------------------------
NOTA SOBRE TAMANO DE SALIDA EN IMAGENES CHICAS (importante)
---------------------------------------------------------------------
Para imagenes chicas, esta version agranda el canvas de trabajo
ANTES de binarizar (interpolacion cubica), y el PNG resultante queda
a ese tamano mas grande -- ya no es pixel-a-pixel igual al de
entrada. Esto es intencional y no es "inventar" detalle: la foto ya
trae, en el borde entre una forma y el papel, un degrade de un par
de pixeles (producto del blur optico + demosaicing de la camara) en
vez de un salto abrupto de un pixel al otro. Al binarizar sobre un
canvas mas grande, ese degrade real se traduce en un borde mas
preciso y menos "escalonado" en vez de forzarse al grosor de un solo
pixel de la foto original. Se probo -- QUE la foto es una posicion
del borde con mas resolucion de muestreo, no una alucinacion de una
red neuronal ni un suavizado cosmetico por fuera de la imagen.

Esto es seguro para el pipeline de lectura porque
read_camera_image_via_markers() usa la salida de scanner_effect()
como la unica imagen de trabajo para TODAS las fases siguientes
(deteccion de markers, quad, homografia, warp final) -- no vuelve a
comparar contra el tamano de la foto original en ningun punto. Igual
se documenta aca explicitamente porque es un cambio de contrato
respecto a versiones anteriores que si preservaban el tamano exacto.
---------------------------------------------------------------------
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Debajo de este tamano (lado mas largo, en px) se usa el camino
# simple + upscale. Elegido con margen por encima de resoluciones
# tipicas de camara chica (480x640, 600x800) y por debajo de fotos
# de documento completo con celular moderno (2000px+).
_SMALL_IMAGE_MAX_DIM = 900

# Tamano de trabajo objetivo (lado mas largo, en px) para el camino
# chico despues del upscale. Cuanto mas chica la foto de entrada,
# mas factor de escala se aplica, hasta el tope _MAX_UPSCALE.
_TARGET_WORKING_DIM = 4032
_MAX_UPSCALE = 8.0


def _percentile_contrast_stretch(gray, low_pct=2, high_pct=98):
    low = np.percentile(gray, low_pct)
    high = np.percentile(gray, high_pct)
    if high <= low:
        return gray
    return np.clip(
        (gray.astype(np.float32) - low) * 255.0 / (high - low), 0, 255
    ).astype(np.uint8)


def estimate_background(gray, close_frac=0.35, down_max=400):
    """
    Estima la iluminacion de fondo (sin las formas oscuras) usando
    un CLOSING morfologico sobre una version reducida de la imagen.
    Ver notas de diseno en el modulo. Solo se usa en el camino
    "imagen grande".
    """
    h, w = gray.shape
    scale = down_max / max(h, w)
    small = cv2.resize(
        gray,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )

    dh, dw = small.shape
    k = max(15, int(min(dh, dw) * close_frac) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    bg_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    bg_small = cv2.GaussianBlur(bg_small, (0, 0), sigmaX=k / 4)

    background = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)
    return background


def _scanner_effect_large(gray):
    """
    Camino completo: corrige iluminacion desigual antes de
    binarizar. Para fotos de documento con margen alrededor del
    codigo.
    """
    background = estimate_background(gray)
    background = np.maximum(background, 1).astype(np.float32)

    normalized = (gray.astype(np.float32) / background) * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    contrast = _percentile_contrast_stretch(normalized)

    _, result = cv2.threshold(
        contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Limpieza de ruido sal-y-pimienta. Seguro aca porque en estas
    # fotos el codigo suele tener bastante resolucion de sobra.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return result


def _scanner_effect_small(gray):
    """
    Camino para imagenes chicas / de cerca:

      1. Upscale cubico ANTES de binarizar (ver nota de modulo) para
         aprovechar el degrade de antialiasing real que ya trae la
         foto, en vez de perderlo contra el threshold.
      2. Sin correccion de fondo, sin limpieza morfologica agresiva
         -- prioriza no perder detalle fino (como el punto de los
         corner markers) por encima de prolijidad cosmetica.
      3. Un afilado leve (unsharp mask) antes del threshold, que
         ayuda a la consistencia de tamano entre markers (se probo
         empiricamente).
    """
    h, w = gray.shape
    scale = min(_MAX_UPSCALE, max(1.0, _TARGET_WORKING_DIM / max(h, w)))

    if scale > 1.0:
        gray = cv2.resize(
            gray,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )

    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    sharp = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)

    contrast = _percentile_contrast_stretch(sharp)

    _, result = cv2.threshold(
        contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return result


def scanner_effect(image):
    # ---------------------------------------------------------
    # 1. Escala de grises + suavizado muy ligero
    # ---------------------------------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    h, w = gray.shape
    if max(h, w) <= _SMALL_IMAGE_MAX_DIM:
        return _scanner_effect_small(gray)
    return _scanner_effect_large(gray)


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

    h, w = image.shape[:2]
    is_small = max(h, w) <= _SMALL_IMAGE_MAX_DIM
    path_used = "chica (upscale + sin correccion de fondo)" if is_small else "grande (con correccion de fondo)"

    result = scanner_effect(image)

    output_path = input_path.with_name(f"{input_path.stem}_scanner.png")
    if not cv2.imwrite(str(output_path), result):
        print(f"Error: no se pudo guardar: {output_path}")
        sys.exit(1)

    print()
    print("Lancherix Scanner Preprocessor")
    print("==============================")
    print(f"Input : {input_path} ({w}x{h})")
    print(f"Output: {output_path} ({result.shape[1]}x{result.shape[0]})")
    print(f"Camino: {path_used}")
    print()


if __name__ == "__main__":
    main()