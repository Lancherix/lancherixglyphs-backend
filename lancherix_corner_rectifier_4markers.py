"""
lancherix_corner_rectifier_4markers.py
=================================

Reescritura desde cero del rectificador de corner markers de Lancherix
Visual Code, con una lógica mucho más simple que las versiones
anteriores (lancherix_corner_rectifier.py / _4markers.py).

La idea (pensada por el autor): en vez de intentar medir los lados
rectos reales de cada marker con Hough y combinarlos en rectas
globales, se aprovecha que los 4 centros de los agujeros blancos YA
se detectan de forma confiable (fase 1, sin cambios), y se construye
todo lo demás con geometría simple sobre esos 4 puntos.

PIPELINE (6 pasos, uno por cada imagen de debug que el script genera):

    PASO 1 -- CANDIDATOS Y CUADRILÁTERO (heredado, sin cambios)
        Detecta los 4 centros blancos (agujeros) de los corner
        markers y arma el cuadrilátero por combinatoria + score
        geométrico. Esta parte ya funciona bien y no se toca.

    PASO 2 -- CUADRILÁTERO DE CENTROS
        Dibuja las líneas que conectan los 4 centros detectados,
        formando el cuadrilátero "crudo" (con la perspectiva real de
        la foto todavía presente).
        -> {stem}_step2_quad.png

    PASO 3 -- CLASIFICAR LADOS LARGOS/CORTOS Y PONER TODA LA IMAGEN
              EN PERSPECTIVA
        Mide los 4 lados del cuadrilátero. Los dos lados más largos
        (en promedio) se consideran "arriba/abajo" y los dos más
        cortos "izquierda/derecha" -- esto es lo que permite
        reconocer el código aunque esté fotografiado de lado (rotado
        90°), porque el marcador es siempre más ancho que alto por
        diseño (relación 3:1 del código completo).
        Con esa clasificación se calcula una homografía que lleva los
        4 centros a un rectángulo (con las proporciones REALMENTE
        medidas, no forzadas a 3:1 -- forzar 3:1 acá introduciría
        error para valores de k chicos, donde el rectángulo de
        centros todavía no se aproxima tanto a 3:1) y esa homografía
        se aplica a TODA la imagen, no solo al rectángulo: el destino
        incluye un margen alrededor para no perder el contexto ni el
        borde real del código, que todavía está más afuera que los
        centros de los markers.
        -> {stem}_step3a_clasificacion.png (qué lado se consideró largo/corto)
        -> {stem}_step3b_warped.png (imagen completa ya en perspectiva)

    PASO 4 -- RECTÁNGULO REAL DEL CÓDIGO
        Ya con la imagen sin perspectiva, el rectángulo de centros de
        markers quedó perfectamente axis-aligned. Se dibuja un segundo
        rectángulo, más grande, paralelo, y a la misma distancia hacia
        afuera en los 4 lados: esa distancia es el radio de un
        glifo/módulo (mitad de su ancho, mitad de su alto), estimado
        proyectando el tamaño de marker ya conocido (fase 1) a través
        de la misma homografía del paso 3.
        -> {stem}_step4_borders.png

    PASO 5 -- CUADRÍCULA Y RECORTE DEFINITIVO
        Se dibuja la cuadrícula de módulos dentro del rectángulo
        exterior (a partir del tamaño de módulo estimado en el paso
        4) y se recorta la imagen a ese rectángulo exterior, dejando
        solo el código.
        -> {stem}_step5_grid_debug.png (con cuadrícula, para inspección)
        -> {stem}_step5_cropped.png (recorte limpio, sin dibujar nada)

    PASO 6 -- CORRECCIÓN FINAL DE ORIENTACIÓN (180°)
        Los pasos 3-5 ya resuelven la ambigüedad de 90°/270° (código
        de lado), pero queda una ambigüedad de 180° posible (código
        boca abajo pero ya "horizontal"). Se mide la orientación
        impresa real del marker que quedó en la esquina superior
        izquierda geométrica (banda con más negro, igual que la fase
        2 de las versiones anteriores) y se compara contra la
        orientación esperada de TL en el diseño (0°, ver
        lancherix_shapes.EXPECTED... / CORNER_MARKER_GLYPHS[0]):
            - si se mide ~0°  -> ya está correcta, no se hace nada.
            - si se mide ~180° -> el código está boca abajo, se rota
              la imagen 180°.
            - cualquier otro valor (90°/270°) indica que la
              clasificación del paso 3 falló; se avisa por consola y
              NO se aplica ninguna rotación automática (rotar 90° acá
              rompería el recorte ya hecho, que asume la proporción
              3:1 correcta).
        -> {stem}_step6_final.png (esta es la imagen que se le pasaría
           al reader)

Uso standalone (debug):
    python3 lancherix_corner_rectifier_v3.py imagen.png
"""

from __future__ import annotations

import sys
import math
import itertools
from pathlib import Path

import cv2
import numpy as np


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# --- Paso 1: detección de agujeros blancos (heredado, sin cambios) --------

MIN_COMPONENT_AREA = 20
MAX_COMPONENT_AREA_FRACTION = 0.01  # fracción del área total de la imagen

MIN_SIZE = 3
MAX_SIZE = 100

MIN_CIRCULARITY = 0.20
MAX_ELLIPSE_ASPECT = 4.5

OPPOSITE_SIDE_TOLERANCE = 0.45
DIAGONAL_RATIO_MIN = 0.35

# El agujero blanco central mide esta fracción del marker completo
# (ver lancherix_shapes.render_glyph). marker_size = hole_size / RATIO.
HOLE_TO_MARKER_RATIO = 0.24

# --- Paso 3: margen alrededor del rectángulo de centros, al poner toda
#     la imagen en perspectiva. En unidades de "módulo" (tamaño de
#     marker estimado en la imagen ORIGINAL). Generoso a propósito:
#     solo tiene que alcanzar para cubrir el borde real del código
#     (paso 4) más algo de contexto; si sobra, se recorta en el paso 5.
MARGIN_MODULES = 3.0

# --- Paso 6: bandas internas para medir la orientación impresa de un
#     marker (heredado de la fase 2 de las versiones anteriores).
BAND_MARGIN = 0.05
BAND_END = 0.45
DARK_THRESHOLD = 100

# Orientación IMPRESA esperada (grados) de cada esquina, según el
# diseño del generador (ver lancherix_shapes.py). Solo TL se usa acá,
# pero se deja el dict completo por claridad / uso futuro.
EXPECTED_PRINTED_ORIENTATION = {
    "TL": 0,
    "TR": 90,
    "BR": 270,
    "BL": 0,
}

EPS = 1e-9

OPPOSITE_LABEL = {"TL": "BR", "TR": "BL", "BR": "TL", "BL": "TR"}


# ============================================================================
# UTILIDADES GEOMÉTRICAS BÁSICAS
# ============================================================================

def distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def polygon_area(points):
    pts = np.asarray(points, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def angle_between(a, b, c):
    ba = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    bc = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < EPS or norm_bc < EPS:
        return 0.0

    cos_value = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_value = np.clip(cos_value, -1.0, 1.0)
    return math.degrees(math.acos(cos_value))


# ============================================================================
# PASO 1 — DETECCIÓN ROBUSTA DE AGUJEROS BLANCOS (heredado, sin cambios)
# ============================================================================

def detect_white_center_candidates(image):
    """
    Detecta los centros blancos de los corner markers. Sin cambios
    respecto a las versiones anteriores -- esta parte ya funciona
    bien.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, white = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), dtype=np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        white, connectivity=8,
    )

    image_area = gray.shape[0] * gray.shape[1]
    max_component_area = image_area * MAX_COMPONENT_AREA_FRACTION

    candidates = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < MIN_COMPONENT_AREA:
            continue
        if area > max_component_area:
            continue
        if w < MIN_SIZE or h < MIN_SIZE:
            continue
        if w > MAX_SIZE or h > MAX_SIZE:
            continue

        aspect = w / float(h)
        if aspect < 0.25 or aspect > 4.0:
            continue

        component_mask = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= EPS:
            continue

        circularity = 4.0 * math.pi * contour_area / (perimeter * perimeter)

        ellipse_ok = False
        ellipse_aspect = None
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (_, _), axes, _ = ellipse
                major = max(float(axes[0]), float(axes[1]))
                minor = min(float(axes[0]), float(axes[1]))
                if minor > EPS:
                    ellipse_aspect = major / minor
                    if ellipse_aspect <= MAX_ELLIPSE_ASPECT:
                        ellipse_ok = True
            except cv2.error:
                ellipse_ok = False

        if circularity < MIN_CIRCULARITY and not ellipse_ok:
            continue

        cx, cy = centroids[i]

        radius = max(w, h) * 1.8
        x0 = max(0, int(cx - radius))
        y0 = max(0, int(cy - radius))
        x1 = min(gray.shape[1], int(cx + radius + 1))
        y1 = min(gray.shape[0], int(cy + radius + 1))

        surrounding = gray[y0:y1, x0:x1]
        if surrounding.size == 0:
            continue

        yy, xx = np.indices(surrounding.shape)
        global_x = xx + x0
        global_y = yy + y0
        dist = np.sqrt((global_x - cx) ** 2 + (global_y - cy) ** 2)

        inner_radius = min(w, h) * 0.9
        outer_pixels = surrounding[dist > inner_radius]
        if len(outer_pixels) == 0:
            continue

        dark_fraction = np.mean(outer_pixels < 100)
        if dark_fraction < 0.40:
            continue

        shape_score = min(circularity / 0.90, 1.0)
        ellipse_score = (
            min(1.0 / max(ellipse_aspect, 1.0) * 1.5, 1.0) if ellipse_ok else 0.0
        )

        min_dimension = float(min(w, h))
        if min_dimension <= 20:
            size_score = 1.0
        elif min_dimension <= 30:
            size_score = 0.90
        elif min_dimension <= 40:
            size_score = 0.65
        elif min_dimension <= 50:
            size_score = 0.35
        else:
            size_score = 0.10

        if area <= 400:
            area_score = 1.0
        elif area <= 700:
            area_score = 0.80
        elif area <= 1000:
            area_score = 0.55
        elif area <= 1500:
            area_score = 0.25
        else:
            area_score = 0.05

        score = (
            0.25 * shape_score + 0.15 * ellipse_score + 0.25 * dark_fraction
            + 0.20 * size_score + 0.15 * area_score
        )

        candidates.append({
            "center": (float(cx), float(cy)),
            "x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area),
            "circularity": float(circularity),
            "dark_fraction": float(dark_fraction),
            "ellipse_aspect": float(ellipse_aspect) if ellipse_aspect is not None else None,
            "score": float(score),
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def order_quad(points):
    pts = np.asarray(points, dtype=np.float32)

    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]

    tl = pts[np.argmin(sums)]
    br = pts[np.argmax(sums)]
    tr = pts[np.argmax(diffs)]
    bl = pts[np.argmin(diffs)]

    return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]


def score_quad(points):
    if len(points) != 4:
        return -1.0

    tl, tr, br, bl = order_quad(points)

    top = distance(tl, tr)
    right = distance(tr, br)
    bottom = distance(br, bl)
    left = distance(bl, tl)

    if min(top, right, bottom, left) < 10:
        return -1.0

    diagonal_1 = distance(tl, br)
    diagonal_2 = distance(tr, bl)

    if min(diagonal_1, diagonal_2) < 10:
        return -1.0

    diagonal_ratio = min(diagonal_1, diagonal_2) / max(diagonal_1, diagonal_2)
    if diagonal_ratio < DIAGONAL_RATIO_MIN:
        return -1.0

    horizontal_ratio = min(top, bottom) / max(top, bottom)
    vertical_ratio = min(left, right) / max(left, right)

    if horizontal_ratio < (1.0 - OPPOSITE_SIDE_TOLERANCE):
        return -1.0
    if vertical_ratio < (1.0 - OPPOSITE_SIDE_TOLERANCE):
        return -1.0

    angles = [
        angle_between(bl, tl, tr),
        angle_between(tl, tr, br),
        angle_between(tr, br, bl),
        angle_between(br, bl, tl),
    ]

    if min(angles) < 35:
        return -1.0
    if max(angles) > 145:
        return -1.0

    angle_error = sum(abs(angle - 90.0) for angle in angles) / 360.0
    side_error = ((1.0 - horizontal_ratio) + (1.0 - vertical_ratio)) / 2.0
    diagonal_error = 1.0 - diagonal_ratio

    return float(
        1.0 - 0.40 * angle_error - 0.35 * side_error - 0.25 * diagonal_error
    )


def find_best_quad(candidates):
    if len(candidates) < 4:
        return None

    best = None
    best_score = -1.0

    for combination in itertools.combinations(candidates, 4):
        points = [candidate["center"] for candidate in combination]

        area = polygon_area(order_quad(points))
        if area < 1000:
            continue

        score = score_quad(points)

        if score > best_score:
            best_score = score
            best = {
                "candidates": combination,
                "points": order_quad(points),
                "score": score,
                "area": area,
            }

    return best


def build_marker_results(best_quad):
    """
    Nota: las etiquetas TL/TR/BR/BL que asigna esta función son
    puramente GEOMÉTRICAS (qué candidato quedó en qué posición de la
    foto), no necesariamente corresponden todavía a la semántica real
    del diseño (eso se resuelve recién en el paso 6).
    """
    if best_quad is None:
        return None

    labels = ["TL", "TR", "BR", "BL"]
    marker_results = []

    for label, point in zip(labels, best_quad["points"]):
        nearest = min(
            best_quad["candidates"],
            key=lambda candidate: distance(candidate["center"], point),
        )
        marker_results.append({**nearest, "label": label})

    return marker_results


def estimate_marker_size(candidate):
    """
    Tamaño estimado del marker completo (= un módulo), a partir del
    tamaño del agujero blanco central y la proporción conocida del
    generador.
    """
    hole_diameter = (candidate["w"] + candidate["h"]) / 2.0
    return hole_diameter / HOLE_TO_MARKER_RATIO


# ============================================================================
# PASO 2 — CUADRILÁTERO DE CENTROS (debug)
# ============================================================================

def draw_quad_debug(image, marker_results):
    debug = image.copy()
    lookup = {m["label"]: m for m in marker_results}
    labels = ("TL", "TR", "BR", "BL")
    points = [lookup[label]["center"] for label in labels]

    pts_int = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(debug, [pts_int], True, (0, 255, 0), 3, cv2.LINE_AA)

    for label, point in zip(labels, points):
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(debug, (x, y), 10, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            debug, label, (x + 14, y - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
        )

    return debug


# ============================================================================
# PASO 3 — LARGO/CORTO Y PERSPECTIVA DE TODA LA IMAGEN
# ============================================================================

def measure_rectangle_dimensions(marker_results):
    """
    Ancho y alto del rectángulo de centros, promediando las dos
    mediciones de cada dimensión (arriba/abajo para el ancho,
    izquierda/derecha para el alto).
    """
    lookup = {m["label"]: m for m in marker_results}

    top = distance(lookup["TL"]["center"], lookup["TR"]["center"])
    bottom = distance(lookup["BL"]["center"], lookup["BR"]["center"])
    left = distance(lookup["TL"]["center"], lookup["BL"]["center"])
    right = distance(lookup["TR"]["center"], lookup["BR"]["center"])

    width = (top + bottom) / 2.0
    height = (left + right) / 2.0

    return width, height


def reorder_markers_by_long_short(marker_results):
    """
    Reclasifica cuál par de lados opuestos es "largo" (arriba/abajo)
    y cuál es "corto" (izquierda/derecha), sin asumir que el
    etiquetado geométrico de la fase 1 ya los puso del lado correcto
    -- el código puede estar fotografiado de lado (rotado 90°).

    Si los lados verticales (izquierda/derecha) resultan ser, en
    promedio, más largos que los horizontales, se rota la asignación
    de etiquetas UN lugar en el ciclo TL->TR->BR->BL->TL, lo que
    intercambia cuál par se llama "arriba/abajo" y cuál
    "izquierda/derecha", sin tocar el orden geométrico real de los
    puntos (siguen siendo las 4 esquinas del mismo cuadrilátero).

    Devuelve (marker_results_reordenado, rotated: bool).
    """
    lookup = {m["label"]: m for m in marker_results}
    ordered = [lookup["TL"], lookup["TR"], lookup["BR"], lookup["BL"]]

    top = distance(ordered[0]["center"], ordered[1]["center"])
    right = distance(ordered[1]["center"], ordered[2]["center"])
    bottom = distance(ordered[2]["center"], ordered[3]["center"])
    left = distance(ordered[3]["center"], ordered[0]["center"])

    horizontal_avg = (top + bottom) / 2.0
    vertical_avg = (left + right) / 2.0

    rotated = vertical_avg > horizontal_avg
    if rotated:
        ordered = ordered[1:] + ordered[:1]

    labels = ("TL", "TR", "BR", "BL")
    new_marker_results = [
        {**candidate, "label": label} for label, candidate in zip(labels, ordered)
    ]
    return new_marker_results, rotated


def draw_side_classification_debug(image, marker_results, rotated):
    """
    Muestra, sobre la imagen original, cuáles lados del cuadrilátero
    de centros se clasificaron como largos (rojo, van a ser
    arriba/abajo) y cuáles como cortos (azul, van a ser
    izquierda/derecha).
    """
    debug = image.copy()
    lookup = {m["label"]: m for m in marker_results}
    tl, tr, br, bl = (lookup[l]["center"] for l in ("TL", "TR", "BR", "BL"))

    def to_int(p):
        return (int(round(p[0])), int(round(p[1])))

    long_color = (0, 0, 255)   # rojo = lado largo -> arriba/abajo
    short_color = (255, 0, 0)  # azul = lado corto -> izquierda/derecha

    cv2.line(debug, to_int(tl), to_int(tr), long_color, 4, cv2.LINE_AA)
    cv2.line(debug, to_int(bl), to_int(br), long_color, 4, cv2.LINE_AA)
    cv2.line(debug, to_int(tl), to_int(bl), short_color, 4, cv2.LINE_AA)
    cv2.line(debug, to_int(tr), to_int(br), short_color, 4, cv2.LINE_AA)

    for label, point in zip(("TL", "TR", "BR", "BL"), (tl, tr, br, bl)):
        x, y = to_int(point)
        cv2.circle(debug, (x, y), 9, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            debug, label, (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA,
        )

    note = "codigo rotado 90 (detectado)" if rotated else "sin rotacion de 90 necesaria"
    cv2.putText(
        debug, note, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA,
    )

    return debug


def compute_whole_image_warp(image, marker_results, margin_modules=MARGIN_MODULES):
    """
    Calcula la homografía que lleva los 4 centros (ya reclasificados
    por reorder_markers_by_long_short) a un rectángulo con las
    proporciones REALMENTE medidas (no forzadas a 3:1), y la aplica a
    TODA la imagen -- no solo al área entre markers -- agregando un
    margen alrededor para no perder el borde real del código (que
    está más afuera que los centros) ni el contexto.
    """
    lookup = {m["label"]: m for m in marker_results}
    width, height = measure_rectangle_dimensions(marker_results)

    if width < EPS or height < EPS:
        return None

    avg_module_size_source = float(
        np.mean([estimate_marker_size(m) for m in marker_results])
    )
    margin_px = int(round(avg_module_size_source * margin_modules))
    width_px = max(int(round(width)), 10)
    height_px = max(int(round(height)), 10)

    src_points = np.array(
        [lookup[l]["center"] for l in ("TL", "TR", "BR", "BL")], dtype=np.float32,
    )

    dst_tl = (float(margin_px), float(margin_px))
    dst_tr = (float(margin_px + width_px), float(margin_px))
    dst_br = (float(margin_px + width_px), float(margin_px + height_px))
    dst_bl = (float(margin_px), float(margin_px + height_px))
    dst_points = np.array([dst_tl, dst_tr, dst_br, dst_bl], dtype=np.float32)

    canvas_w = width_px + 2 * margin_px
    canvas_h = height_px + 2 * margin_px

    H = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(
        image, H, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR,
    )

    return {
        "H": H,
        "warped": warped,
        "dst_points": {"TL": dst_tl, "TR": dst_tr, "BR": dst_br, "BL": dst_bl},
        "canvas_size": (canvas_w, canvas_h),
        "width_px": width_px, "height_px": height_px, "margin_px": margin_px,
    }


# ============================================================================
# PASO 4 — RECTÁNGULO REAL DEL CÓDIGO
# ============================================================================

def estimate_module_size_in_destination(marker_results, H):
    """
    Proyecta, a través de la misma homografía del paso 3, el tamaño
    de módulo ya conocido (fase 1, por marker) para estimar cuánto
    mide un módulo en la imagen YA rectificada. Se promedia sobre los
    4 markers y sobre las direcciones X e Y por separado (podrían
    quedar levemente distintas si el escalado de la homografía no es
    perfectamente isotrópico).
    """
    widths = []
    heights = []

    for marker in marker_results:
        cx, cy = marker["center"]
        module_size_source = estimate_marker_size(marker)

        pts_source = np.array([
            [cx, cy],
            [cx + module_size_source, cy],
            [cx, cy + module_size_source],
        ], dtype=np.float32).reshape(-1, 1, 2)

        pts_dest = cv2.perspectiveTransform(pts_source, H).reshape(-1, 2)
        center_dest, x_dest, y_dest = pts_dest

        widths.append(distance(center_dest, x_dest))
        heights.append(distance(center_dest, y_dest))

    return float(np.mean(widths)), float(np.mean(heights))


def compute_k_and_module_size(width_px, height_px):
    """
    Deduce k (numero de filas de modulos) y el tamano de modulo real
    a partir del rectangulo de centros YA rectificado (sin
    perspectiva, ver compute_whole_image_warp). El layout del
    generador es un invariante conocido:

        width_px  (centro a centro, horizontal) = (3k - 1) * module_size
        height_px (centro a centro, vertical)   = (k - 1)  * module_size

    Esto reemplaza la estimacion de estimate_module_size_in_destination
    (que depende de proyectar el tamano fisico del marker a traves de
    la homografia, y es ruidosa) por un calculo exacto basado solo en
    las distancias ya medidas entre centros.
    """
    if height_px < EPS:
        return None

    ratio = width_px / height_px
    denominator = ratio - 3.0
    if abs(denominator) < EPS:
        return None

    k_float = (ratio - 1.0) / denominator
    k = int(round(k_float))
    if k < 2:
        return None

    module_w = width_px / (3 * k - 1)
    module_h = height_px / (k - 1)

    return {
        "k": k,
        "k_float": float(k_float),
        "module_w": float(module_w),
        "module_h": float(module_h),
        "module_size": float((module_w + module_h) / 2.0),
    }


def compute_outer_rectangle(dst_points, module_w, module_h):
    """
    Rectángulo del borde REAL del código: paralelo al rectángulo de
    centros, más grande, y a una distancia uniforme hacia afuera en
    los 4 lados igual al radio de un módulo (mitad de su ancho, mitad
    de su alto).
    """
    margin_x = module_w / 2.0
    margin_y = module_h / 2.0

    tl = (dst_points["TL"][0] - margin_x, dst_points["TL"][1] - margin_y)
    tr = (dst_points["TR"][0] + margin_x, dst_points["TR"][1] - margin_y)
    br = (dst_points["BR"][0] + margin_x, dst_points["BR"][1] + margin_y)
    bl = (dst_points["BL"][0] - margin_x, dst_points["BL"][1] + margin_y)

    return {"TL": tl, "TR": tr, "BR": br, "BL": bl}


def draw_borders_debug(warped, dst_points, outer_points):
    debug = warped.copy()

    inner_pts = np.array(
        [dst_points[l] for l in ("TL", "TR", "BR", "BL")], dtype=np.int32,
    ).reshape(-1, 1, 2)
    outer_pts = np.array(
        [outer_points[l] for l in ("TL", "TR", "BR", "BL")], dtype=np.int32,
    ).reshape(-1, 1, 2)

    cv2.polylines(debug, [inner_pts], True, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.polylines(debug, [outer_pts], True, (255, 0, 0), 3, cv2.LINE_AA)

    return debug


# ============================================================================
# PASO 5 — CUADRÍCULA Y RECORTE DEFINITIVO
# ============================================================================

def crop_to_outer_rectangle(warped_image, outer_points):
    height, width = warped_image.shape[:2]

    x0 = max(0, int(round(outer_points["TL"][0])))
    y0 = max(0, int(round(outer_points["TL"][1])))
    x1 = min(width, int(round(outer_points["BR"][0])))
    y1 = min(height, int(round(outer_points["BR"][1])))

    if x1 <= x0 or y1 <= y0:
        return None, (0, 0)

    cropped = warped_image[y0:y1, x0:x1].copy()
    return cropped, (x0, y0)


def draw_grid_debug(cropped, module_w, module_h, k=None):
    debug = cropped.copy()
    height, width = debug.shape[:2]

    if k is not None:
        cols = 3 * k
        rows = k
    else:
        cols = max(1, int(round(width / module_w))) if module_w > EPS else 1
        rows = max(1, int(round(height / module_h))) if module_h > EPS else 1

    for c in range(cols + 1):
        x = int(round(c * width / cols))
        cv2.line(debug, (x, 0), (x, height - 1), (0, 255, 0), 1, cv2.LINE_AA)

    for r in range(rows + 1):
        y = int(round(r * height / rows))
        cv2.line(debug, (0, y), (width - 1, y), (0, 255, 0), 1, cv2.LINE_AA)

    return debug


# ============================================================================
# PASO 6 — CORRECCIÓN FINAL DE ORIENTACIÓN (180°)
# ============================================================================

def locate_tl_marker_in_cropped(dst_points, crop_offset):
    x0, y0 = crop_offset
    tl = dst_points["TL"]
    return (tl[0] - x0, tl[1] - y0)

def locate_marker_in_cropped(dst_points, crop_offset, label):
    x0, y0 = crop_offset
    point = dst_points[label]
    return (point[0] - x0, point[1] - y0)

def measure_printed_orientation(image, center, marker_size):
    """
    Mide, con el mismo método de bandas de las versiones anteriores
    (fase 2), la orientación impresa (0/90/180/270) del marker
    centrado en `center` con tamaño `marker_size`.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cx, cy = center
    half = marker_size / 2.0
    outer = half * BAND_MARGIN
    inner = half * BAND_END

    def measure_region(x0, y0, x1, y1):
        x0 = max(0, int(math.floor(x0)))
        y0 = max(0, int(math.floor(y0)))
        x1 = min(gray.shape[1], int(math.ceil(x1)))
        y1 = min(gray.shape[0], int(math.ceil(y1)))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        region = gray[y0:y1, x0:x1]
        if region.size == 0:
            return 0.0
        return float(np.count_nonzero(region < DARK_THRESHOLD) / region.size)

    left = measure_region(cx - half + outer, cy - half + outer, cx - half + inner, cy + half - outer)
    right = measure_region(cx + half - inner, cy - half + outer, cx + half - outer, cy + half - outer)
    top = measure_region(cx - half + outer, cy - half + outer, cx + half - outer, cy - half + inner)
    bottom = measure_region(cx - half + outer, cy + half - inner, cx + half - outer, cy + half - outer)

    measurements = {"left": left, "top": top, "right": right, "bottom": bottom}
    max_side = max(measurements, key=measurements.get)

    # máximo derecha -> 0°, máximo abajo -> 90°, máximo izquierda -> 180°, máximo arriba -> 270°
    orientation_map = {"right": 0, "bottom": 90, "left": 180, "top": 270}
    return orientation_map[max_side], measurements


def determine_final_rotation(cropped, dst_points, crop_offset, avg_module_size):
    """
    IMPORTANTE: EXPECTED_PRINTED_ORIENTATION no es simetrica respecto
    a un giro de 180 grados (TL=0 y BR=270 -- no son "opuestos"
    ciclicos por +180). Esto significa que, si la foto esta
    realmente al reves, el marker que geometricamente cae en la
    posicion TL del canvas (etiquetado por order_quad segun
    coordenadas puras) es en realidad, fisicamente, el marker que el
    generador dibujo en la esquina BR -- la diagonalmente opuesta --
    y su orientacion medida sera (EXPECTED[BR] + 180) % 360, NO
    (EXPECTED[TL] + 180) % 360.

    Por eso se evaluan DOS hipotesis por separado para cada marker
    (sin rotar / rotado 180, comparando contra la esquina opuesta) y
    se vota cual de las dos explica mejor las 4 mediciones, en vez de
    exigir diff==0 o diff==180 contra la misma etiqueta.
    """
    labels = ("TL", "TR", "BR", "BL")
    measurements = {}
    votes = {0: 0, 180: 0}

    for label in labels:
        center = locate_marker_in_cropped(dst_points, crop_offset, label)
        measured, bands = measure_printed_orientation(cropped, center, avg_module_size)

        diff_no_rotation = (measured - EXPECTED_PRINTED_ORIENTATION[label]) % 360
        opposite = OPPOSITE_LABEL[label]
        diff_180_rotation = (
            measured - EXPECTED_PRINTED_ORIENTATION[opposite] - 180
        ) % 360

        measurements[label] = {
            "measured": measured, "bands": bands,
            "diff_no_rotation": diff_no_rotation,
            "diff_180_rotation": diff_180_rotation,
        }

        if diff_no_rotation == 0 and diff_180_rotation != 0:
            votes[0] += 1
        elif diff_180_rotation == 0 and diff_no_rotation != 0:
            votes[180] += 1
        # si ninguna (o ambas) calzan exacto, no cuenta como voto --
        # ese marker se midio con ruido.

    total_votes = votes[0] + votes[180]
    winner = 180 if votes[180] > votes[0] else 0

    if total_votes >= 3 and votes[winner] >= 3:
        detail = (
            f"Consenso {votes[winner]}/{total_votes} a favor de "
            f"rotacion={winner}. mediciones={measurements}"
        )
        return winner, detail, measurements

    detail = (
        f"Sin consenso suficiente para decidir (votos: {votes}, "
        f"total evaluable: {total_votes}/4). mediciones={measurements}"
    )
    return None, detail, measurements


def apply_final_orientation(cropped, measured_orientation):
    """
    Compara la orientación medida del marker TL geométrico contra la
    esperada (0°) y aplica, como mucho, una rotación de 180°. Si la
    medición da 90° o 270°, algo falló antes (paso 3): no se corrige
    a ciegas, porque una rotación de 90° acá invalidaría el recorte
    ya hecho (que asume el ancho/alto ya correctos).
    """
    expected = EXPECTED_PRINTED_ORIENTATION["TL"]
    diff = (measured_orientation - expected) % 360

    if diff == 0:
        return cropped, 0, "orientacion correcta, no se aplico rotacion"
    if diff == 180:
        return cv2.rotate(cropped, cv2.ROTATE_180), 180, "boca abajo detectado, se roto 180"

    return (
        cropped, 0,
        f"ORIENTACION INESPERADA ({measured_orientation} grados, se esperaba 0 o 180); "
        f"no se aplico ninguna correccion -- revisar la clasificacion del paso 3"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) != 2:
        print("Uso:\n  python3 lancherix_corner_rectifier_v3.py imagen.png")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"ERROR: archivo no encontrado: {input_path}")
        sys.exit(1)

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"ERROR: no se pudo abrir: {input_path}")
        sys.exit(1)

    def out_path(suffix):
        return input_path.parent / f"{input_path.stem}_{suffix}.png"

    print()
    print("Lancherix Corner Rectifier v3 (logica simplificada)")
    print("=====================================================")
    print(f"Input: {input_path}")

    # ---------------------------------------------------------------
    # PASO 1
    # ---------------------------------------------------------------
    candidates = detect_white_center_candidates(image)
    print(f"\nCandidatos encontrados: {len(candidates)}")

    best_quad = find_best_quad(candidates)
    if best_quad is None:
        print("NO SE ENCONTRO UN CUADRILATERO PLAUSIBLE. Abortando.")
        sys.exit(1)

    marker_results = build_marker_results(best_quad)
    print(f"Cuadrilatero aproximado, score={best_quad['score']:.4f}")

    # ---------------------------------------------------------------
    # PASO 2
    # ---------------------------------------------------------------
    step2_debug = draw_quad_debug(image, marker_results)
    cv2.imwrite(str(out_path("step2_quad")), step2_debug)
    print(f"Debug paso 2 generado: {out_path('step2_quad')}")

    # ---------------------------------------------------------------
    # PASO 3
    # ---------------------------------------------------------------
    marker_results_ordered, rotated = reorder_markers_by_long_short(marker_results)
    print(f"\nClasificacion largo/corto: rotacion de 90 {'SI' if rotated else 'NO'} detectada")

    step3a_debug = draw_side_classification_debug(image, marker_results_ordered, rotated)
    cv2.imwrite(str(out_path("step3a_clasificacion")), step3a_debug)
    print(f"Debug paso 3a generado: {out_path('step3a_clasificacion')}")

    warp_info = compute_whole_image_warp(image, marker_results_ordered)
    if warp_info is None:
        print("ERROR: no se pudo calcular la homografia (rectangulo degenerado). Abortando.")
        sys.exit(1)

    cv2.imwrite(str(out_path("step3b_warped")), warp_info["warped"])
    print(f"Debug paso 3b generado: {out_path('step3b_warped')}")
    print(
        f"  Rectangulo de centros medido: {warp_info['width_px']}x{warp_info['height_px']}px "
        f"(margen: {warp_info['margin_px']}px)"
    )

    # ---------------------------------------------------------------
    # PASO 4
    # ---------------------------------------------------------------
    k_info = compute_k_and_module_size(warp_info["width_px"], warp_info["height_px"])

    if k_info is not None:
        module_w, module_h = k_info["module_w"], k_info["module_h"]
        print(f"\nk deducido = {k_info['k']} (estimado {k_info['k_float']:.3f})")
        print(f"Tamano de modulo (exacto, via k): {module_w:.2f}x{module_h:.2f}px")
    else:
        # fallback: si la geometria de centros no da un k valido
        # (deberia ser raro), se vuelve al metodo anterior.
        module_w, module_h = estimate_module_size_in_destination(
            marker_results_ordered, warp_info["H"],
        )
        print(f"\nADVERTENCIA: no se pudo deducir k, usando estimacion por marker")
        print(f"Tamano de modulo estimado (fallback): {module_w:.2f}x{module_h:.2f}px")

    outer_points = compute_outer_rectangle(warp_info["dst_points"], module_w, module_h)
    step4_debug = draw_borders_debug(warp_info["warped"], warp_info["dst_points"], outer_points)
    cv2.imwrite(str(out_path("step4_borders")), step4_debug)
    print(f"Debug paso 4 generado: {out_path('step4_borders')}")

    # ---------------------------------------------------------------
    # PASO 5
    # ---------------------------------------------------------------
    cropped, crop_offset = crop_to_outer_rectangle(warp_info["warped"], outer_points)
    if cropped is None:
        print("ERROR: el rectangulo exterior quedo fuera de la imagen. Abortando.")
        sys.exit(1)

    grid_debug = draw_grid_debug(
        cropped, module_w, module_h,
        k=(k_info["k"] if k_info is not None else None),
    )
    cv2.imwrite(str(out_path("step5_grid_debug")), grid_debug)
    cv2.imwrite(str(out_path("step5_cropped")), cropped)
    print(f"Debug paso 5 generado: {out_path('step5_grid_debug')}")
    print(f"Recorte limpio generado: {out_path('step5_cropped')}")
    print(f"  Tamano recortado: {cropped.shape[1]}x{cropped.shape[0]}px")

    # ---------------------------------------------------------------
    # PASO 6
    # ---------------------------------------------------------------
    avg_module_size = (module_w + module_h) / 2.0

    rotation_needed, orientation_note, band_measurements = determine_final_rotation(
        cropped, warp_info["dst_points"], crop_offset, avg_module_size,
    )

    if rotation_needed == 180:
        final_image = cv2.rotate(cropped, cv2.ROTATE_180)
    else:
        final_image = cropped
        rotation_needed = 0

    cv2.imwrite(str(out_path("step6_final")), final_image)
    print(f"\nOrientacion (paso 6, consenso de 4 markers):")
    for label, info in band_measurements.items():
        print(
            f"  {label}: medido={info['measured']} "
            f"diff_sin_rot={info['diff_no_rotation']} "
            f"diff_180={info['diff_180_rotation']} "
            f"bandas={info['bands']}"
        )
    print(f"  {orientation_note}")
    print(f"  Rotacion aplicada: {rotation_needed} grados")
    print(f"Final (para el reader) generado: {out_path('step6_final')}")


if __name__ == "__main__":
    main()