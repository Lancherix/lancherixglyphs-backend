"""
lancherix_camera_reader.py
============================

Pruebas a realizar:

   python3 lancherix_generator.py test-normal "hola" test_normal.png
   python3 lancherix_camera_reader.py test_normal.png

   python3 lancherix_generator.py test-perspective "hola" test_persp_corto.png
   python3 lancherix_camera_reader.py test_persp_corto.png

   python3 lancherix_generator.py test-perspective "https://ejemplo.com/una-url-bastante-larga-para-probar-k-grande" test_persp_largo.png
   python3 lancherix_camera_reader.py test_persp_largo.png

Localiza y decodifica un Lancherix Visual Code en una imagen con
perspectiva real (posicion, escala, rotacion y perspectiva
arbitrarias), usando los 4 corner markers.

IMPORTANTE (version consolidada, con codec ECC):

Este archivo contenia originalmente DOS pipelines de localizacion:

  1. read_camera_image (viejo, YA REMOVIDO): localizaba el codigo
     probando muchos candidatos de cuadrilatero genericos + muchos
     valores posibles de k, evaluando cada uno con un score de
     plantillas, y resolviendo la correspondencia de vertices
     probando 8 hipotesis de rotacion/reflexion. No necesitaba los
     corner markers, pero era lento y con riesgo de falsos positivos.

  2. read_camera_image_via_markers (el unico que queda): usa los 4
     corner markers de lancherix_corner_rectifier_4markers.py. A
     diferencia de la version anterior con solo 3 markers (TL/TR/BL),
     que necesitaba inferir el cuarto corner con un punto de fuga y
     una pasada de refinamiento (refine_bottom_anchor), con los 4
     corners detectados DIRECTAMENTE cada uno de los 4 lados del
     codigo se mide dos veces (una desde cada marker que lo toca) y
     se combina en una unica recta por minimos cuadrados. El quad
     exacto sale de intersectar esas 4 rectas -- sin puntos de fuga,
     sin pasadas de refinamiento -- y k sale de forma directa a
     partir de las distancias, ya rectificadas, entre los 4 centros.

Si en el futuro se necesita volver a soportar imagenes SIN corner
markers, conviene revivir el pipeline (1) desde el historial en vez
de reconstruirlo desde cero.

CAMBIO RESPECTO A LA VERSION ANTERIOR (sin ECC):

La decodificacion final ya NO mapea glifo -> texto de forma directa
(module_values_to_text, formato viejo sin correccion de errores).
Ahora sigue el flujo de lancherix_shapes.py:

    leer la grilla completa -> excluir las 3 celdas de esquina (por
    indice: 0, row_width - 1, (num_rows - 1) * row_width) -> buscar
    el indice de END_GLYPH -> tomar todo lo anterior -> mapear cada
    glifo a su valor con GLYPH_TO_VALUE -> decode_symbols(values,
    num_ecc_symbols)

`num_ecc_symbols` debe ser el MISMO valor usado al generar el codigo
(DEFAULT_ECC_SYMBOLS de lancherix_shapes.py, salvo que el generador
se haya invocado con un valor distinto explicito).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from lancherix_shapes import (
    QUIET_ZONE,
    GRID_ASPECT_RATIO,
    END_GLYPH,
    GLYPH_TO_VALUE,
    DEFAULT_ECC_SYMBOLS,
    decode_symbols,
    _dominant_colors,
    _resolve_background_foreground,
)

from lancherix_scanner_preprocessor import scanner_effect

from lancherix_corner_rectifier_4markers import (
    detect_white_center_candidates,
    find_best_quad,
    build_marker_results,
    detect_marker_geometries,
    build_precise_quad,
    estimate_outer_corners_fallback,
    rectify_from_corners,
    rectify_marker_centers,
    compute_k_and_module_size,
    draw_precise_debug,
    draw_grid_debug,
    analyze_four_markers,
    detect_orientation_correction,
    _label_permutation,
)


# --------------------------------------------------------------------------
# CONFIGURACION
# --------------------------------------------------------------------------

# Tamano de modulo (px) del canvas canonico final. Mismo valor que
# _TEMPLATE_SIZE en lancherix_shapes.py (decision de arquitectura #4)
# -- se pasa explicitamente a _resolve_background_foreground para
# que el template matching interno use el mismo tamano.
_TEMPLATE_SIZE = 40


# --------------------------------------------------------------------------
# DECODIFICACION DE LA IMAGEN CANONICA (saltando las 3 esquinas)
# --------------------------------------------------------------------------

def _decode_canonical_image(
    canonical_rgb: np.ndarray,
    row_width: int,
    num_rows: int,
    module_size: float = _TEMPLATE_SIZE,
    num_ecc_symbols: int = DEFAULT_ECC_SYMBOLS,
) -> str:
    """
    Decodifica la imagen ya canonicalizada (warp aplicado, orientacion
    ya resuelta via corner markers) a texto, usando el codec CON
    correccion de errores.

    La grilla se recorre fila por fila, columna por columna. Se
    excluyen por INDICE las 3 celdas reservadas para corner_marker
    (TL, TR, BL -- ver layout_glyph_rows_with_markers en
    lancherix_shapes.py). La celda BR NO esta reservada: es un glyph
    de datos normal mas y se incluye en la decodificacion.

    Se corta la secuencia en el primer END_GLYPH encontrado (entre
    las celdas ya filtradas); todo lo que venga despues (relleno
    decorativo) se ignora. Los glifos restantes se mapean a sus
    valores de 3 bits y se pasan a decode_symbols(), que reconstruye
    el texto corrigiendo errores con Reed-Solomon cuando es posible.
    """
    color_a, color_b = _dominant_colors(canonical_rgb)
    _bg, _fg, _gray, glyph_sequence = _resolve_background_foreground(
        canonical_rgb, color_a, color_b, row_width, num_rows, module_size,
        template_size=module_size,
    )

    corner_indices = {0, row_width - 1, (num_rows - 1) * row_width}
    filtered = [g for i, g in enumerate(glyph_sequence) if i not in corner_indices]

    try:
        end_index = filtered.index(END_GLYPH)
    except ValueError:
        raise ValueError(
            "No se encontro el marcador de FIN. La imagen puede estar "
            "danada, mal localizada, o no ser un Lancherix Visual Code "
            "valido."
        )

    data_glyphs = filtered[:end_index]

    try:
        values = [GLYPH_TO_VALUE[g] for g in data_glyphs]
    except KeyError as exc:
        raise ValueError(
            f"Se encontro un glifo de datos invalido: {exc}. La imagen "
            "puede estar danada o mal localizada."
        )

    return decode_symbols(values, num_ecc_symbols)


# --------------------------------------------------------------------------
# PIPELINE COMPLETO: localizacion + decodificacion via corner markers
# --------------------------------------------------------------------------

def read_camera_image_via_markers(
    image_path: str,
    num_ecc_symbols: int = DEFAULT_ECC_SYMBOLS,
) -> str:
    """
    Localiza un Lancherix Visual Code en una imagen con perspectiva
    real usando los 4 corner markers, y decodifica el texto (con
    correccion de errores Reed-Solomon).

    A diferencia de un enfoque por fuerza bruta (probar muchos k y
    muchas correspondencias de vertices), este pipeline es directo:

      1. Detectar los 4 agujeros blancos centrales de los corner
         markers y armar el cuadrilatero TL/TR/BR/BL aproximado
         (combinatoria + score geometrico) -- sin ambiguedad de
         "cual esquina es cual".
      2. Extraer los lados rectos reales de cada marker (Hough), ya
         en coordenadas globales.
      3. Combinar, para cada uno de los 4 lados del codigo, las dos
         mediciones de los markers que lo tocan en una unica recta
         (minimos cuadrados), e intersectar esas 4 rectas para
         obtener el quad EXACTO. No hace falta punto de fuga ni
         pasada de refinamiento: con los 4 corners detectados
         directamente, cada lado ya tiene evidencia real de sobra.
      4. Si esa deteccion de lineas no reune evidencia suficiente
         (imagen muy borrosa/pequena), caer de vuelta al metodo mas
         simple que estima el borde exterior a partir del tamano del
         agujero blanco.
      5. Deducir k directamente a partir de las distancias
         rectificadas entre los 4 centros (promediando las dos
         mediciones de cada distancia, sin probar candidatos).
      6. Warpear la imagen original a un canvas canonico k x 3k (con
         quiet zone) usando el quad final, y decodificar con el codec
         CON ECC (decode_symbols).

    `num_ecc_symbols` debe coincidir con el valor usado al generar el
    codigo (DEFAULT_ECC_SYMBOLS salvo que se haya generado con otro
    valor explicito).

    Guarda en disco PNGs de debug (`_quad.png` con las lineas Hough y
    el quad exacto, `_rectified.png`, `_grid.png` con la cuadricula
    deducida), mas un canvas canonico final
    (`_k{k}_canonical_via_markers.png`).
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo abrir la imagen: {image_path}")

    # Aplica el efecto scanner (blancos/negros puros) directo sobre la
    # foto original, sin necesidad de un archivo intermedio en disco.
    scanned = scanner_effect(image)
    if len(scanned.shape) == 2:
        image = cv2.cvtColor(scanned, cv2.COLOR_GRAY2BGR)
    else:
        image = scanned.copy()
    cv2.imwrite(f"{Path(image_path).stem}_scanner.png", scanned)

    # ------------------------------------------------------------------
    # FASE 1 -- candidatos y cuadrilatero aproximado (TL/TR/BR/BL).
    # ------------------------------------------------------------------
    candidates = detect_white_center_candidates(image)
    best_quad = find_best_quad(candidates)

    if best_quad is None:
        raise ValueError(
            "No se pudieron identificar los 4 corner markers "
            "(TL/TR/BR/BL) en la imagen."
        )

    marker_results = build_marker_results(best_quad)

    # ------------------------------------------------------------------
    # FASE 2/3 -- lados reales (Hough) + cuadrilatero exacto por
    # interseccion. Si falla, cae al fallback por tamano de agujero.
    # ------------------------------------------------------------------
    marker_geoms = detect_marker_geometries(image, marker_results)
    precise_result = build_precise_quad(marker_geoms)

    if precise_result is not None:
        corners = precise_result["corners"]
    else:
        corners = estimate_outer_corners_fallback(marker_results)
        if corners is None:
            raise ValueError(
                "No se pudo construir el cuadrilatero (ni por lineas "
                "reales ni por el metodo de respaldo) a partir de los "
                "corner markers detectados."
            )

    # ------------------------------------------------------------------
    # FASE 2 -- bandas internas + deteccion de rotacion/espejo.
    #
    # `corners` (de precise_result o del fallback) esta etiquetado por
    # POSICION en la imagen (el punto mas arriba-izquierda se llama
    # "TL"), no por identidad fisica del marker impreso. Si el codigo
    # esta rotado 180 grados en la foto, ese punto "TL" por posicion
    # es en realidad la esquina impresa como BR. Hay que remapear las
    # etiquetas a la identidad fisica real antes de usar `corners`
    # para el warp canonico final, o el codigo se lee empezando por
    # la esquina equivocada.
    # ------------------------------------------------------------------
    phase2_results = analyze_four_markers(image, marker_results)
    orientation_correction = detect_orientation_correction(phase2_results)
    label_map = _label_permutation(orientation_correction)  # {pos_label: physical_label}

    corners = {label_map[pos_label]: point for pos_label, point in corners.items()}

    print()
    print("ORIENTATION CORRECTION")
    print("-----------------------")
    print(f"  {orientation_correction['detail']}")
    print(
        f"  rotacion={orientation_correction['rotation']} "
        f"espejo={orientation_correction['mirror']}"
    )

    # ------------------------------------------------------------------
    # FASE 4 -- rectificacion (proporcion 3:1 fija) + debug.
    # ------------------------------------------------------------------
    rectified, homography = rectify_from_corners(image, corners)

    if rectified is None:
        raise ValueError(
            "No se pudo rectificar la imagen a partir del cuadrilatero "
            "detectado."
        )

    input_name = Path(image_path).stem

    quad_debug_path = f"{input_name}_quad.png"
    cv2.imwrite(quad_debug_path, draw_precise_debug(image, marker_geoms, precise_result))
    print()
    print("QUAD IMAGE SAVED")
    print("----------------")
    print(quad_debug_path)

    rectified_debug_path = f"{input_name}_rectified.png"
    cv2.imwrite(rectified_debug_path, rectified)
    print()
    print("RECTIFIED IMAGE SAVED")
    print("---------------------")
    print(rectified_debug_path)

    # ------------------------------------------------------------------
    # FASE 5 -- k YA CONOCIDO: se deduce directo, sin probar candidatos.
    # ------------------------------------------------------------------
    rectified_centers = rectify_marker_centers(marker_results, homography)
    k_info = compute_k_and_module_size(rectified_centers)

    if k_info is None:
        raise ValueError(
            "No se pudo deducir k (num_rows) a partir de los corner "
            "markers rectificados."
        )

    k = k_info["k"]

    grid_debug_path = f"{input_name}_grid.png"
    cv2.imwrite(grid_debug_path, draw_grid_debug(rectified, rectified_centers, k_info))
    print()
    print("GRID IMAGE SAVED")
    print("----------------")
    print(grid_debug_path)
    print()
    print(f"k = {k} (estimado {k_info['k_float']:.3f})")
    print(f"modulo estimado = {k_info['module_size']:.2f}px")

    row_width = GRID_ASPECT_RATIO * k

    # ------------------------------------------------------------------
    # Warp FINAL directo: imagen original -> canvas canonico CON quiet
    # zone (mismo formato que espera _decode_canonical_image), en un
    # solo paso, usando el quad final (en coordenadas de la imagen
    # ORIGINAL -- no las de `rectified`, que ya perdio la quiet zone y
    # esta en otra escala/tamano).
    # ------------------------------------------------------------------
    module_size = _TEMPLATE_SIZE

    canvas_w = int(round((row_width + 2 * QUIET_ZONE) * module_size))
    canvas_h = int(round((k + 2 * QUIET_ZONE) * module_size))

    margin_x = QUIET_ZONE * module_size
    margin_y = QUIET_ZONE * module_size

    dst = np.array([
        [margin_x, margin_y],
        [canvas_w - 1 - margin_x, margin_y],
        [canvas_w - 1 - margin_x, canvas_h - 1 - margin_y],
        [margin_x, canvas_h - 1 - margin_y],
    ], dtype=np.float32)

    src = np.array([
        corners["TL"],
        corners["TR"],
        corners["BR"],
        corners["BL"],
    ], dtype=np.float32)

    final_matrix = cv2.getPerspectiveTransform(src, dst)

    canonical_bgr = cv2.warpPerspective(
        image, final_matrix, (canvas_w, canvas_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )

    canonical_path = f"{input_name}_k{k}_canonical_via_markers.png"
    cv2.imwrite(canonical_path, canonical_bgr)

    print()
    print("CANONICAL IMAGE SAVED (via corner markers)")
    print("-------------------------------------------")
    print(canonical_path)

    canonical_rgb = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2RGB)

    text = _decode_canonical_image(
        canonical_rgb, row_width, k, module_size, num_ecc_symbols,
    )

    print()
    print(f"k = {k}")

    return text


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Uso: python lancherix_camera_reader.py "
            "<ruta_a_imagen.png>"
        )
        sys.exit(1)

    try:
        text = read_camera_image_via_markers(sys.argv[1])
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}")
        sys.exit(1)

    print(f"Texto decodificado: {text}")