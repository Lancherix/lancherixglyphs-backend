"""
lancherix_corner_rectifier_4markers.py
======================================

Localiza los 4 corner markers (TL, TR, BR, BL) de un Lancherix Visual
Code en una imagen con perspectiva real arbitraria, y construye la
homografía que rectifica el código a un rectángulo limpio.

Con los 4 corners detectados directamente (a diferencia de
lancherix_corner_rectifier.py, que solo tiene 3 y necesita inferir el
cuarto con un punto de fuga) el problema se simplifica: cada uno de
los 4 lados del código (arriba, abajo, izquierda, derecha) puede
medirse DOS VECES -- una desde cada uno de los dos markers que lo
tocan -- y esas dos mediciones se combinan en una única recta global
por mínimos cuadrados. Los cuatro corners finales salen de
intersectar esas cuatro rectas entre sí. Esto es exacto bajo
cualquier perspectiva, porque no depende de asumir ningún tamaño o
simetría del marker: solo depende de encontrar sus lados rectos
reales.

FASES:

    FASE 1 -- CANDIDATOS Y CUADRILÁTERO APROXIMADO
        Detecta los cuatro centros blancos (agujeros) de los corner
        markers con un detector robusto (tamaño, forma, elipse,
        entorno oscuro) y arma el cuadrilátero TL/TR/BR/BL por
        combinatoria + score geométrico (igual que antes).

    FASE 2 -- BANDAS INTERNAS (heredada, sin cambios de fondo)
        Mide la cantidad de negro en cuatro bandas internas de cada
        marker. Sirve como experimento/verificación de la orientación
        con la que fue impreso cada marker.

    FASE 3 -- LADOS REALES POR MARKER (Hough) [NUEVO]
        Por cada uno de los 4 markers, extrae un patch a su alrededor
        y corre Hough para encontrar sus dos lados rectos exteriores,
        ya en coordenadas GLOBALES de la imagen (igual que hace
        lancherix_corner_rectifier.py con sus 3 markers).

    FASE 4 -- CUADRILÁTERO EXACTO POR INTERSECCIÓN [NUEVO]
        Combina las dos mediciones de cada lado del código
        (superior: TL+TR, inferior: BR+BL, izquierdo: TL+BL, derecho:
        TR+BR) en una única recta robusta (ajuste por mínimos
        cuadrados), e intersecta esas 4 rectas para obtener los 4
        corners EXACTOS del código -- ya no estimados a partir de un
        tamaño de marker asumido, como hacía la versión anterior de
        este archivo.

    FASE 5 -- RECTIFICACIÓN
        Homografía de los 4 corners exactos a un rectángulo, y
        warpPerspective. Si la fase 3/4 no logra detectar líneas
        suficientes (imagen muy borrosa o de baja resolución), se cae
        de vuelta al método anterior, más simple, que estima el borde
        exterior a partir del tamaño del agujero blanco.

Uso standalone (debug):
    python3 lancherix_corner_rectifier_4markers.py imagen.png
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

# --- Fase 1: detección de agujeros blancos -----------------------------

MIN_COMPONENT_AREA = 20
MAX_COMPONENT_AREA_FRACTION = 0.01  # fracción del área total de la imagen

MIN_SIZE = 3
MAX_SIZE = 100

MIN_CIRCULARITY = 0.20
MAX_ELLIPSE_ASPECT = 4.5

# --- Fase 1: cuadrilátero -------------------------------------------------

OPPOSITE_SIDE_TOLERANCE = 0.45
DIAGONAL_RATIO_MIN = 0.35

# --- Geometría del marker --------------------------------------------------

# El agujero blanco central mide esta fracción del marker completo
# (ver lancherix_shapes.render_glyph). marker_size = hole_size / RATIO.
HOLE_TO_MARKER_RATIO = 0.24

# --- Fase 2: bandas internas ------------------------------------------------

BAND_MARGIN = 0.05
BAND_END = 0.45
DARK_THRESHOLD = 100

# --- Fase 3: extracción de líneas (Hough) -----------------------------------

PATCH_PADDING = 0.35

HOUGH_THRESHOLD = 25
HOUGH_MIN_LINE_LENGTH = 10
HOUGH_MAX_LINE_GAP = 6

# --- Debug -------------------------------------------------------------

DEBUG_RADIUS = 18

# El código completo (código + los 3k-1 x (k-1) módulos) siempre tiene
# una proporción ancho:alto de 3:1 (ver lancherix_shapes). Esto es un
# invariante del generador, no algo que se deba medir en la imagen.
OUTPUT_ASPECT_RATIO = 3.0

EPS = 1e-9

# --- Fase 5b: corrección de orientación (rotación / espejo) ----------------

# Orientación IMPRESA esperada (grados) de cada esquina, según el diseño
# del generador (ver lancherix_shapes.py, asignación esquina->orientación).
EXPECTED_PRINTED_ORIENTATION = {
    "TL": 0,
    "TR": 90,
    "BR": 0,
    "BL": 90,
}

# ============================================================================
# UTILIDADES GEOMÉTRICAS BÁSICAS
# ============================================================================

def distance(a, b):
    return math.hypot(
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1]),
    )


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


# ----- rectas (representadas como ax + by + c = 0, con (a,b) unitario) -----

def cross(a, b):
    return np.cross(a, b)


def line_from_points(p1, p2):
    p1h = np.asarray([p1[0], p1[1], 1.0], dtype=np.float64)
    p2h = np.asarray([p2[0], p2[1], 1.0], dtype=np.float64)

    l = cross(p1h, p2h)
    n = math.hypot(l[0], l[1])
    if n < EPS:
        return None
    return l / n


def line_angle(line):
    a, b, _ = line
    angle = math.degrees(math.atan2(-a, b))
    angle %= 180.0
    return angle


def angle_difference(a, b):
    d = abs(a - b) % 180.0
    if d > 90.0:
        d = 180.0 - d
    return d


def intersect_lines(l1, l2):
    if l1 is None or l2 is None:
        return None
    p = cross(l1, l2)
    if abs(p[2]) < EPS:
        return None
    return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float64)


def fit_global_line(items):
    """
    Ajusta una única recta (por mínimos cuadrados) a través de los
    puntos extremos de una o más mediciones (items con "p1"/"p2" en
    coordenadas GLOBALES). Se usa para combinar, por ejemplo, la
    medición del lado superior hecha desde TL con la medición del
    mismo lado superior hecha desde TR.
    """
    points = []
    for item in items:
        points.append(item["p1"])
        points.append(item["p2"])

    if len(points) < 2:
        return None

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)

    vx, vy, x0, y0 = cv2.fitLine(
        pts,
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    ).flatten()

    a, b = float(vy), float(-vx)
    norm = math.hypot(a, b)
    if norm < EPS:
        return None

    a, b = a / norm, b / norm
    c = -(a * x0 + b * y0)

    return np.array([a, b, c], dtype=np.float64)


# ============================================================================
# FASE 1 — DETECCIÓN ROBUSTA DE AGUJEROS BLANCOS
# ============================================================================

def detect_white_center_candidates(image):
    """
    Detecta los centros blancos de los corner markers.

    El centro real de cada marker es un pequeño agujero blanco dentro
    de una forma negra. Bajo perspectiva puede convertirse en una
    elipse, así que la circularidad no se usa como requisito
    estricto: se combina tamaño del componente, aspect ratio,
    circularidad, ajuste de elipse, entorno predominantemente oscuro
    y penalización de regiones blancas demasiado grandes, para no
    confundir zonas blancas grandes del contenido del código con
    centros de marker.
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


# ============================================================================
# ORDENAMIENTO Y SCORE DEL CUADRILÁTERO (candidatos -> TL/TR/BR/BL)
# ============================================================================

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
    Asigna a cada uno de los 4 puntos ordenados (TL, TR, BR, BL) el
    candidato original más cercano, y devuelve una lista de
    diccionarios "marker" con una clave "label" añadida.
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
    Tamaño estimado del marker completo, a partir del tamaño del
    agujero blanco central y de la proporción conocida del generador.
    """
    hole_diameter = (candidate["w"] + candidate["h"]) / 2.0
    return hole_diameter / HOLE_TO_MARKER_RATIO


# ============================================================================
# FASE 2 — BANDAS INTERNAS (heredada)
# ============================================================================

def analyze_marker_inner_zones(image, candidate):
    """
    Divide el marker en cuatro BANDAS completas (arriba/abajo cubren
    todo el ancho, izquierda/derecha cubren todo el alto) y mide la
    fracción de píxeles oscuros en cada una. Sirve para verificar con
    qué orientación fue impreso cada marker.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cx, cy = candidate["center"]
    marker_size = estimate_marker_size(candidate)
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

        dark_pixels = np.count_nonzero(region < DARK_THRESHOLD)
        return float(dark_pixels / region.size)

    left = measure_region(
        cx - half + outer, cy - half + outer,
        cx - half + inner, cy + half - outer,
    )
    right = measure_region(
        cx + half - inner, cy - half + outer,
        cx + half - outer, cy + half - outer,
    )
    top = measure_region(
        cx - half + outer, cy - half + outer,
        cx + half - outer, cy - half + inner,
    )
    bottom = measure_region(
        cx - half + outer, cy + half - inner,
        cx + half - outer, cy + half - outer,
    )

    return {"left": left, "top": top, "right": right, "bottom": bottom}


def analyze_four_markers(image, marker_results):
    """
    Mide las cuatro bandas de cada marker y deduce, con qué
    orientación fue impreso (0/90/180/270), a partir de cuál banda
    concentra más negro.

        máximo derecha  -> marker 0°
        máximo abajo    -> marker 90°
        máximo izquierda -> marker 180°
        máximo arriba   -> marker 270°
    """
    if not marker_results:
        return None

    orientation_map = {"right": 0, "bottom": 90, "left": 180, "top": 270}

    results = {}

    for marker in marker_results:
        measurements = analyze_marker_inner_zones(image, marker)
        max_side = max(measurements, key=measurements.get)
        printed_orientation = orientation_map[max_side]

        results[marker["label"]] = {
            **measurements,
            "max_side": max_side,
            "printed_orientation": printed_orientation,
        }

    return results


# ============================================================================
# FASE 3 — LADOS RECTOS REALES DE CADA MARKER (Hough)
# ============================================================================

def extract_marker_patch(image, candidate):
    """
    Extrae un patch alrededor del corner marker completo, a partir
    del agujero blanco detectado (diámetro del agujero =
    HOLE_TO_MARKER_RATIO * tamaño del marker completo).
    """
    h, w = image.shape[:2]
    cx, cy = candidate["center"]

    marker_size = estimate_marker_size(candidate)
    pad = marker_size * PATCH_PADDING
    half_size = marker_size / 2.0 + pad

    x0 = max(0, int(round(cx - half_size)))
    y0 = max(0, int(round(cy - half_size)))
    x1 = min(w, int(round(cx + half_size)))
    y1 = min(h, int(round(cy + half_size)))

    return image[y0:y1, x0:x1], (x0, y0)


def extract_lines_from_patch(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 120)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180.0,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    result = []
    if lines is None:
        return result

    for item in lines:
        x1, y1, x2, y2 = map(float, item[0]) if item.ndim == 2 else map(float, item)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < HOUGH_MIN_LINE_LENGTH:
            continue

        line = line_from_points((x1, y1), (x2, y2))
        if line is None:
            continue

        result.append({
            "p1": np.array([x1, y1]), "p2": np.array([x2, y2]),
            "line": line, "angle": line_angle(line), "length": length,
        })

    return result


def transform_line_to_global(local_line, offset):
    a, b, c = local_line
    ox, oy = offset

    result = np.array([a, b, c - a * ox - b * oy], dtype=np.float64)
    n = math.hypot(result[0], result[1])
    if n < EPS:
        return result
    return result / n


def choose_marker_side_lines(candidate, image, vertical_side, horizontal_side):
    """
    Detecta los dos lados exteriores rectos de un corner marker, ya
    en coordenadas GLOBALES:

        vertical_side:   "left" o "right"
        horizontal_side: "top" o "bottom"

    Por ejemplo TL usa ("left", "top"), BR usa ("right", "bottom").

    A diferencia de la primera versión, esto NO hace una votación
    "a ciegas" sobre todas las líneas del patch (lo cual, en un
    código con contenido denso pegado al marker, terminaba eligiendo
    bordes de las figuras del contenido en vez del propio borde del
    marker). En cambio, usa como prior la posición donde SABEMOS que
    debería estar ese lado -- a partir del tamaño estimado del marker
    (candidate["center"] +/- half, ver estimate_marker_size) -- y solo
    acepta líneas de Hough que caigan cerca de esa posición esperada
    y que además sean razonablemente rectas en la dirección correcta
    (bien horizontales para el lado de arriba/abajo, bien verticales
    para el de izquierda/derecha). Se devuelve la LISTA completa de
    coincidencias (no solo la "mejor"), porque el borde real suele
    aparecer partido en varios segmentos de Hough (interrumpidos por
    el agujero blanco o por el contenido); esa lista completa se
    ajusta luego por mínimos cuadrados en fit_global_line().
    """
    patch, offset = extract_marker_patch(image, candidate)
    if patch is None or patch.size == 0:
        return None

    local_lines = extract_lines_from_patch(patch)
    if not local_lines:
        return None

    offset_vec = np.asarray(offset, dtype=np.float64)

    lines = []
    for item in local_lines:
        global_line = transform_line_to_global(item["line"], offset)
        lines.append({
            **item,
            "line": global_line,
            "p1": item["p1"] + offset_vec,
            "p2": item["p2"] + offset_vec,
        })

    cx, cy = candidate["center"]
    half = estimate_marker_size(candidate) / 2.0

    expected_y = (cy - half) if horizontal_side == "top" else (cy + half)
    expected_x = (cx - half) if vertical_side == "left" else (cx + half)

    def find_matches(steepness_ratio, position_of, expected_position, tolerance):
        matches = []
        for item in lines:
            dx = abs(item["p2"][0] - item["p1"][0])
            dy = abs(item["p2"][1] - item["p1"][1])

            if position_of == "y":
                # Buscamos líneas casi HORIZONTALES: dx debe dominar sobre dy.
                if dx < steepness_ratio * dy:
                    continue
                position = (item["p1"][1] + item["p2"][1]) / 2.0
            else:
                # Buscamos líneas casi VERTICALES: dy debe dominar sobre dx.
                if dy < steepness_ratio * dx:
                    continue
                position = (item["p1"][0] + item["p2"][0]) / 2.0

            if abs(position - expected_position) <= tolerance:
                matches.append(item)

        return matches

    base_tolerance = max(half * 0.5, 6.0)

    horizontal_matches = find_matches(2.0, "y", expected_y, base_tolerance)
    if not horizontal_matches:
        # Segundo intento, más permisivo, por si la perspectiva o el
        # ruido corrieron el borde un poco más de lo esperado.
        horizontal_matches = find_matches(1.5, "y", expected_y, base_tolerance * 2.5)

    vertical_matches = find_matches(2.0, "x", expected_x, base_tolerance)
    if not vertical_matches:
        vertical_matches = find_matches(1.5, "x", expected_x, base_tolerance * 2.5)

    return {
        "horizontal": horizontal_matches or None,
        "vertical": vertical_matches or None,
    }


# ============================================================================
# FASE 4 — CUADRILÁTERO EXACTO POR INTERSECCIÓN DE LOS 4 LADOS
# ============================================================================

# Qué lado vertical/horizontal le corresponde a cada esquina.
CORNER_SIDES = {
    "TL": ("left", "top"),
    "TR": ("right", "top"),
    "BR": ("right", "bottom"),
    "BL": ("left", "bottom"),
}

# Qué dos markers contribuyen a cada uno de los 4 lados del código.
EDGE_CONTRIBUTORS = {
    "top": (("TL", "horizontal"), ("TR", "horizontal")),
    "bottom": (("BR", "horizontal"), ("BL", "horizontal")),
    "left": (("TL", "vertical"), ("BL", "vertical")),
    "right": (("TR", "vertical"), ("BR", "vertical")),
}


def detect_marker_geometries(image, marker_results):
    """
    Corre choose_marker_side_lines() para cada uno de los 4 markers.
    Devuelve un dict {label: geometry|None}.
    """
    marker_geoms = {}

    for marker in marker_results:
        label = marker["label"]
        vertical_side, horizontal_side = CORNER_SIDES[label]
        marker_geoms[label] = choose_marker_side_lines(
            marker, image, vertical_side, horizontal_side,
        )

    return marker_geoms


def build_precise_quad(marker_geoms):
    """
    Combina, para cada uno de los 4 lados del código, las mediciones
    de los dos markers que lo tocan en una única recta (mínimos
    cuadrados), e intersecta esas 4 rectas para obtener los 4 corners
    exactos.

    Devuelve None si no hay evidencia suficiente para alguno de los
    4 lados o alguno de los 4 corners.
    """
    edge_lines = {}

    for edge_name, contributors in EDGE_CONTRIBUTORS.items():
        items = []
        for label, key in contributors:
            geometry = marker_geoms.get(label)
            if geometry and geometry.get(key):
                items.extend(geometry[key])

        edge_lines[edge_name] = fit_global_line(items) if items else None

    if any(edge_lines[name] is None for name in ("top", "bottom", "left", "right")):
        return None

    corners = {
        "TL": intersect_lines(edge_lines["top"], edge_lines["left"]),
        "TR": intersect_lines(edge_lines["top"], edge_lines["right"]),
        "BR": intersect_lines(edge_lines["bottom"], edge_lines["right"]),
        "BL": intersect_lines(edge_lines["bottom"], edge_lines["left"]),
    }

    if any(point is None for point in corners.values()):
        return None

    return {"corners": corners, "edges": edge_lines}


# ============================================================================
# FASE 5 — RECTIFICACIÓN
# ============================================================================

def rectify_from_corners(image, corners):
    """
    Rectifica la imagen dado un dict {"TL":.., "TR":.., "BR":.., "BL":..}
    de puntos en coordenadas de imagen.

    IMPORTANTE: el ancho y el alto de salida NO se calculan por
    separado a partir de las distancias top/bottom (para el ancho) y
    left/right (para el alto) del cuadrilátero proyectado. Bajo
    perspectiva esas distancias están foreshortened de forma
    DISTINTA entre sí (por ejemplo el lado más cercano a la cámara
    se ve más largo que el lado más lejano), así que tomar el máximo
    de cada par por separado no reconstruye la proporción real del
    código -- eso fue lo que causaba la escala incorrecta.

    En cambio, la proporción ancho:alto del código es un invariante
    conocido del generador (OUTPUT_ASPECT_RATIO = 3:1), así que se fija
    directamente: solo se estima una resolución de salida razonable
    (a partir del alto aparente izquierdo/derecho) y el ancho sale de
    multiplicar por 3. La homografía ya se encarga de corregir toda la
    distorsión de perspectiva independientemente de esa elección de
    resolución.
    """
    tl = np.asarray(corners["TL"], dtype=np.float32)
    tr = np.asarray(corners["TR"], dtype=np.float32)
    br = np.asarray(corners["BR"], dtype=np.float32)
    bl = np.asarray(corners["BL"], dtype=np.float32)

    left_height = distance(tl, bl)
    right_height = distance(tr, br)

    height = int(round(max(left_height, right_height)))
    width = int(round(height * OUTPUT_ASPECT_RATIO))

    if width < 10 or height < 10:
        return None, None

    source_points = np.array([tl, tr, br, bl], dtype=np.float32)
    destination_points = np.array(
        [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    rectified = cv2.warpPerspective(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR,
    )

    return rectified, matrix


def estimate_outer_corners_fallback(marker_results):
    """
    Método anterior (aproximado): a partir de los 4 centros
    detectados y del tamaño estimado de cada marker (asumiendo que el
    marker es simétrico alrededor de su agujero), se desplaza cada
    centro hacia afuera a lo largo de los lados del cuadrilátero.

    Se usa solo como fallback cuando la fase 3/4 (líneas reales de
    Hough) no logra reunir evidencia suficiente -- por ejemplo en
    imágenes muy borrosas o de muy baja resolución -- porque, a
    diferencia de la intersección de lados reales, este método
    asume un tamaño de marker uniforme y por lo tanto es menos
    preciso bajo perspectiva fuerte.
    """
    lookup = {m["label"]: m for m in marker_results}
    if not all(label in lookup for label in ("TL", "TR", "BR", "BL")):
        return None

    tl_c = np.asarray(lookup["TL"]["center"], dtype=np.float64)
    tr_c = np.asarray(lookup["TR"]["center"], dtype=np.float64)
    br_c = np.asarray(lookup["BR"]["center"], dtype=np.float64)
    bl_c = np.asarray(lookup["BL"]["center"], dtype=np.float64)

    marker_sizes = {
        label: estimate_marker_size(lookup[label])
        for label in ("TL", "TR", "BR", "BL")
    }

    top_vector = tr_c - tl_c
    bottom_vector = br_c - bl_c
    left_vector = bl_c - tl_c
    right_vector = br_c - tr_c

    lengths = [
        np.linalg.norm(top_vector), np.linalg.norm(bottom_vector),
        np.linalg.norm(left_vector), np.linalg.norm(right_vector),
    ]
    if min(lengths) < EPS:
        return None

    top_unit = top_vector / lengths[0]
    bottom_unit = bottom_vector / lengths[1]
    left_unit = left_vector / lengths[2]
    right_unit = right_vector / lengths[3]

    tl_r = marker_sizes["TL"] / 2.0
    tr_r = marker_sizes["TR"] / 2.0
    br_r = marker_sizes["BR"] / 2.0
    bl_r = marker_sizes["BL"] / 2.0

    tl_outer = tl_c - top_unit * tl_r - left_unit * tl_r
    tr_outer = tr_c + top_unit * tr_r - right_unit * tr_r
    br_outer = br_c + bottom_unit * br_r + right_unit * br_r
    bl_outer = bl_c - bottom_unit * bl_r + left_unit * bl_r

    return {
        "TL": tuple(tl_outer), "TR": tuple(tr_outer),
        "BR": tuple(br_outer), "BL": tuple(bl_outer),
    }


# ============================================================================
# FASE 5b — CORRECCIÓN DE ORIENTACIÓN (rotación 90/180/270° y espejo)
# ============================================================================
#
# La fase 2 mide la orientación IMPRESA real de cada marker. Esta fase
# compara esa medición contra la orientación ESPERADA de cada esquina
# (EXPECTED_PRINTED_ORIENTATION) para deducir si la imagen ya
# rectificada (fase 5) quedó rotada y/o espejada respecto a como fue
# generada, y la corrige.

def _closest_multiple_of_90(degrees):
    return int(round(degrees / 90.0) * 90) % 360


def _rotation_offset(measured, expected):
    return _closest_multiple_of_90((measured - expected) % 360)


# Cómo se permutan las etiquetas de esquina (posición geométrica) al
# aplicar cada transformación sobre la imagen rectificada.
_MIRROR_LABEL_MAP = {"TL": "TR", "TR": "TL", "BL": "BR", "BR": "BL"}

_ROTATE_LABEL_MAPS = {
    0:   {"TL": "TL", "TR": "TR", "BR": "BR", "BL": "BL"},
    90:  {"TL": "TR", "TR": "BR", "BR": "BL", "BL": "TL"},
    180: {"TL": "BR", "TR": "BL", "BR": "TL", "BL": "TR"},
    270: {"TL": "BL", "TR": "TL", "BR": "TR", "BL": "BR"},
}


def detect_orientation_correction(phase2_results):
    """
    Devuelve un dict describiendo la corrección a aplicar sobre la
    imagen ya rectificada:

        {
            "kind": "none" | "rotation" | "mirror_rotation" | "unknown",
            "rotation": 0 | 90 | 180 | 270,  # rotación horaria a aplicar
            "mirror": bool,                   # flip horizontal, ANTES de rotar
            "detail": str,
        }

    "unknown" = fase 2 incompleta, o las 4 mediciones no son
    consistentes con ninguna rotación/espejo puro. En ese caso no se
    aplica ninguna corrección.
    """
    labels = ("TL", "TR", "BR", "BL")

    if not phase2_results or not all(label in phase2_results for label in labels):
        return {
            "kind": "unknown", "rotation": 0, "mirror": False,
            "detail": "Fase 2 incompleta: no hay medición en las 4 esquinas.",
        }

    measured = {
        label: phase2_results[label]["printed_orientation"] for label in labels
    }

    # --- Caso 1: rotación pura (sin espejo) ---------------------------------
    offsets = {
        label: _rotation_offset(measured[label], EXPECTED_PRINTED_ORIENTATION[label])
        for label in labels
    }

    if len(set(offsets.values())) == 1:
        rotation = next(iter(offsets.values()))
        return {
            "kind": "rotation" if rotation != 0 else "none",
            "rotation": rotation,
            "mirror": False,
            "detail": (
                f"Rotacion pura detectada: {rotation}. "
                f"medido={measured} esperado={EXPECTED_PRINTED_ORIENTATION}"
            ),
        }

    # --- Caso 2: espejado (+ posible rotación) ------------------------------
    # Un flip horizontal invierte izquierda/derecha (TL<->TR, BL<->BR) y
    # además invierte el sentido en que se lee el ángulo impreso de cada
    # marker (una forma dibujada a A° se ve, espejada, a (360-A)%360).
    mirrored_measured = {
        "TL": (180 - measured["TR"]) % 360,
        "TR": (180 - measured["TL"]) % 360,
        "BL": (180 - measured["BR"]) % 360,
        "BR": (180 - measured["BL"]) % 360,
    }

    mirror_offsets = {
        label: _rotation_offset(mirrored_measured[label], EXPECTED_PRINTED_ORIENTATION[label])
        for label in labels
    }

    if len(set(mirror_offsets.values())) == 1:
        rotation = next(iter(mirror_offsets.values()))
        return {
            "kind": "mirror_rotation",
            "rotation": rotation,
            "mirror": True,
            "detail": f"Espejo detectado (+ rotacion {rotation}). medido={measured}",
        }

    # --- Caso 3: no hay patrón consistente -----------------------------------
    return {
        "kind": "unknown",
        "rotation": 0,
        "mirror": False,
        "detail": (
            f"Sin correccion consistente. medido={measured} "
            f"offsets={offsets} mirror_offsets={mirror_offsets}"
        ),
    }


def _label_permutation(correction):
    """
    Combina el mapa de espejo y el de rotación en {label_original:
    label_final}, en el mismo orden en que se aplican las
    transformaciones sobre la imagen (mirror primero, rotate después).
    """
    rotate_map = _ROTATE_LABEL_MAPS[correction["rotation"]]

    if correction["mirror"]:
        return {
            label: rotate_map[_MIRROR_LABEL_MAP[label]]
            for label in ("TL", "TR", "BR", "BL")
        }
    return {label: rotate_map[label] for label in ("TL", "TR", "BR", "BL")}


def _transform_point(point, width, height, mirror, rotation):
    """
    Transforma un punto (x, y) con las mismas operaciones que
    cv2.flip / cv2.rotate aplican a una imagen width x height.
    """
    x, y = point

    if mirror:
        x = (width - 1) - x

    if rotation == 0:
        return (x, y)
    if rotation == 90:
        return ((height - 1) - y, x)
    if rotation == 180:
        return ((width - 1) - x, (height - 1) - y)
    if rotation == 270:
        return (y, (width - 1) - x)

    raise ValueError(f"rotacion invalida: {rotation}")


def apply_orientation_correction(rectified, rectified_centers, correction):
    """
    Aplica sobre la imagen ya rectificada (y sobre los centros de los
    markers ya rectificados) el flip horizontal y/o la rotación de
    90/180/270 grados detectados por detect_orientation_correction,
    dejando el resultado en la orientación canónica del generador
    (TL=0, TR=90, BR=180, BL=270).

    Devuelve (rectified_corregida, rectified_centers_corregidos).
    """
    if correction["kind"] not in ("rotation", "mirror_rotation"):
        return rectified, rectified_centers

    height, width = rectified.shape[:2]
    mirror = correction["mirror"]
    rotation = correction["rotation"]

    corrected_image = rectified
    if mirror:
        corrected_image = cv2.flip(corrected_image, 1)
    if rotation == 90:
        corrected_image = cv2.rotate(corrected_image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        corrected_image = cv2.rotate(corrected_image, cv2.ROTATE_180)
    elif rotation == 270:
        corrected_image = cv2.rotate(corrected_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    label_map = _label_permutation(correction)
    corrected_centers = {}
    if rectified_centers:
        for old_label, point in rectified_centers.items():
            new_label = label_map[old_label]
            corrected_centers[new_label] = _transform_point(
                point, width, height, mirror, rotation,
            )

    return corrected_image, corrected_centers


# ============================================================================
# FASE 6 — DEDUCIR k A PARTIR DE LA CUADRÍCULA RECTIFICADA
# ============================================================================

def rectify_marker_centers(marker_results, homography):
    """
    Transforma los 4 centros de los agujeros blancos (no los corners
    exteriores) a través de la homografía de rectificación, para
    medir las distancias reales entre markers ya sin perspectiva.
    """
    lookup = {m["label"]: m for m in marker_results}
    labels = ("TL", "TR", "BR", "BL")

    points = np.array(
        [lookup[label]["center"] for label in labels], dtype=np.float32,
    ).reshape(-1, 1, 2)

    rectified = cv2.perspectiveTransform(points, homography).reshape(-1, 2)

    return {label: tuple(point) for label, point in zip(labels, rectified)}


def compute_k_and_module_size(rectified_centers):
    """
    Deduce k (número de filas de módulos) a partir de las distancias,
    ya rectificadas, entre los centros de los 4 corner markers.

    El layout del generador coloca los markers de forma que:

        distancia horizontal (centro a centro) = (3k - 1) módulos
        distancia vertical   (centro a centro) = (k - 1) módulos

    Con las 4 esquinas disponibles hay DOS mediciones independientes
    de cada distancia (arriba/abajo para la horizontal, izquierda/
    derecha para la vertical), así que se promedian para mayor
    robustez -- a diferencia de la versión de 3 markers, que solo
    tenía una medición de cada una.
    """
    tl = rectified_centers["TL"]
    tr = rectified_centers["TR"]
    br = rectified_centers["BR"]
    bl = rectified_centers["BL"]

    horizontal_distance = (distance(tl, tr) + distance(bl, br)) / 2.0
    vertical_distance = (distance(tl, bl) + distance(tr, br)) / 2.0

    if vertical_distance <= EPS:
        return None

    ratio = horizontal_distance / vertical_distance
    denominator = ratio - 3.0
    if abs(denominator) < EPS:
        return None

    k_float = (ratio - 1.0) / denominator
    k = int(round(k_float))
    if k < 2:
        return None

    module_width = horizontal_distance / (3 * k - 1)
    module_height = vertical_distance / (k - 1)
    module_size = (module_width + module_height) / 2.0

    return {
        "k": k,
        "k_float": float(k_float),
        "module_size": float(module_size),
        "horizontal_distance": float(horizontal_distance),
        "vertical_distance": float(vertical_distance),
    }


def draw_grid_debug(rectified_image, rectified_centers, k_info):
    """
    Dibuja, sobre la imagen ya rectificada, la cuadrícula de 3k x k
    módulos deducida, y marca los 4 centros de marker rectificados.
    """
    debug = rectified_image.copy()
    height, width = debug.shape[:2]
    k = k_info["k"]

    for column in range(3 * k + 1):
        x = int(round(column * width / (3 * k)))
        cv2.line(debug, (x, 0), (x, height - 1), (0, 255, 0), 1, cv2.LINE_AA)

    for row in range(k + 1):
        y = int(round(row * height / k))
        cv2.line(debug, (0, y), (width - 1, y), (0, 255, 0), 1, cv2.LINE_AA)

    for label, point in rectified_centers.items():
        x = int(round(point[0]))
        y = int(round(point[1]))
        cv2.circle(debug, (x, y), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            debug, label, (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
        )

    return debug


# ============================================================================
# DEBUG — IMAGEN GENERAL (candidatos + cuadrilátero aproximado + bandas)
# ============================================================================

def draw_overview_debug(image, candidates, marker_results, phase2_results, fallback_corners):
    debug = image.copy()

    for index, candidate in enumerate(candidates):
        cx, cy = candidate["center"]
        center = (int(round(cx)), int(round(cy)))

        cv2.circle(debug, center, 8, (0, 200, 255), 2)
        cv2.putText(
            debug, f"C{index}", (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA,
        )

    if not marker_results:
        return debug

    # Cuadrilátero aproximado (verde), en el orden TL,TR,BR,BL.
    points = [m["center"] for m in marker_results]
    pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(debug, [pts], True, (0, 255, 0), 3, cv2.LINE_AA)

    # Borde exterior estimado (fallback), en azul.
    if fallback_corners is not None:
        outer_pts = np.asarray(
            [fallback_corners["TL"], fallback_corners["TR"],
             fallback_corners["BR"], fallback_corners["BL"]],
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        cv2.polylines(debug, [outer_pts], True, (255, 0, 0), 4, cv2.LINE_AA)
        for point in outer_pts.reshape(-1, 2):
            cv2.circle(debug, tuple(point), 7, (255, 0, 0), -1, cv2.LINE_AA)

    for marker in marker_results:
        x, y = marker["center"]
        x = int(round(x))
        y = int(round(y))
        label = marker["label"]

        cv2.circle(debug, (x, y), DEBUG_RADIUS, (0, 255, 0), 3)

        marker_size = estimate_marker_size(marker)
        half = marker_size / 2.0

        x0 = int(round(x - half))
        y0 = int(round(y - half))
        x1 = int(round(x + half))
        y1 = int(round(y + half))
        cv2.rectangle(debug, (x0, y0), (x1, y1), (0, 255, 255), 2, cv2.LINE_AA)

        outer = half * BAND_MARGIN
        inner = half * BAND_END

        # izquierda / derecha (todo el alto), arriba / abajo (todo el ancho)
        cv2.rectangle(
            debug,
            (int(round(x - half + outer)), int(round(y - half + outer))),
            (int(round(x - half + inner)), int(round(y + half - outer))),
            (255, 0, 255), 2,
        )
        cv2.rectangle(
            debug,
            (int(round(x + half - inner)), int(round(y - half + outer))),
            (int(round(x + half - outer)), int(round(y + half - outer))),
            (255, 0, 255), 2,
        )
        cv2.rectangle(
            debug,
            (int(round(x - half + outer)), int(round(y - half + outer))),
            (int(round(x + half - outer)), int(round(y - half + inner))),
            (255, 0, 255), 2,
        )
        cv2.rectangle(
            debug,
            (int(round(x - half + outer)), int(round(y + half - inner))),
            (int(round(x + half - outer)), int(round(y + half - outer))),
            (255, 0, 255), 2,
        )

        text = label
        if phase2_results and label in phase2_results:
            text += f" ({phase2_results[label]['printed_orientation']}°)"

        cv2.putText(
            debug, text, (x + 22, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2, cv2.LINE_AA,
        )

    return debug


# ============================================================================
# DEBUG — IMAGEN DE LÍNEAS PRECISAS (fase 3/4)
# ============================================================================

def _draw_full_line(debug, line, color, thickness=3):
    height, width = debug.shape[:2]
    a, b, c = line

    if abs(b) > abs(a):
        x0 = 0.0
        y0 = -(a * x0 + c) / b
        x1 = float(width - 1)
        y1 = -(a * x1 + c) / b
    else:
        y0 = 0.0
        x0 = -(b * y0 + c) / a
        y1 = float(height - 1)
        x1 = -(b * y1 + c) / a

    cv2.line(
        debug,
        (int(round(x0)), int(round(y0))),
        (int(round(x1)), int(round(y1))),
        color, thickness, cv2.LINE_AA,
    )


def draw_precise_debug(image, marker_geoms, precise_result):
    """
    Dibuja, en coordenadas globales: los segmentos de Hough elegidos
    por cada marker (cian), las 4 rectas combinadas del código
    (verde grueso) y los 4 corners exactos (rojo), si se pudieron
    calcular.
    """
    debug = image.copy()

    for label, geometry in marker_geoms.items():
        if geometry is None:
            continue
        for key in ("horizontal", "vertical"):
            matches = geometry.get(key)
            if not matches:
                continue
            for item in matches:
                p1 = tuple(int(round(v)) for v in item["p1"])
                p2 = tuple(int(round(v)) for v in item["p2"])
                cv2.line(debug, p1, p2, (255, 255, 0), 3, cv2.LINE_AA)

    if precise_result is None:
        return debug

    for line in precise_result["edges"].values():
        _draw_full_line(debug, line, (0, 255, 0), 3)

    for label, point in precise_result["corners"].items():
        x = int(round(point[0]))
        y = int(round(point[1]))
        cv2.circle(debug, (x, y), 9, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            debug, label, (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
        )

    return debug


# ============================================================================
# REPORTE
# ============================================================================

def print_report(candidates, best_quad, marker_results, phase2_results,
                  precise_result, fallback_corners, used_method, k_info):

    print()
    print("=" * 60)
    print("LANCHERIX 4-MARKER DETECTOR (con perspectiva)")
    print("=" * 60)

    print()
    print(f"Candidatos encontrados: {len(candidates)}")
    for index, candidate in enumerate(candidates):
        cx, cy = candidate["center"]
        print(
            f"  C{index}: center=({cx:.2f}, {cy:.2f}) "
            f"area={candidate['area']} score={candidate['score']:.3f}"
        )

    print()
    if best_quad is None:
        print("NO SE ENCONTRO UN CUADRILATERO PLAUSIBLE.")
        return

    print("CUADRILATERO APROXIMADO (centros de los agujeros)")
    print("---------------------------------------------------")
    for marker in marker_results:
        x, y = marker["center"]
        print(f"  {marker['label']} = ({x:.2f}, {y:.2f})")
    print(f"  Area:  {best_quad['area']:.2f}")
    print(f"  Score: {best_quad['score']:.4f}")

    if phase2_results:
        print()
        print("ORIENTACION IMPRESA DE LOS MARKERS (fase 2, banda con más negro)")
        print("-------------------------------------------------------------------")
        for label in ("TL", "TR", "BR", "BL"):
            result = phase2_results.get(label)
            if result is None:
                continue
            print(
                f"  {label}: izquierda={result['left']:.3f} arriba={result['top']:.3f} "
                f"derecha={result['right']:.3f} abajo={result['bottom']:.3f} "
                f"-> mayor={result['max_side']} "
                f"=> impreso a {result['printed_orientation']}°"
            )

    print()
    print("CUADRILATERO EXACTO (fase 3/4, intersección de lados reales)")
    print("----------------------------------------------------------------")
    if precise_result is not None:
        for label in ("TL", "TR", "BR", "BL"):
            x, y = precise_result["corners"][label]
            print(f"  {label} = ({x:.2f}, {y:.2f})")
    else:
        print("  No se pudo construir (evidencia de líneas insuficiente).")

    print()
    print(f"Método usado para rectificar: {used_method}")

    if used_method == "fallback" and fallback_corners is not None:
        print()
        print("CUADRILATERO ESTIMADO (fallback, a partir del tamaño del agujero)")
        print("-----------------------------------------------------------------")
        for label in ("TL", "TR", "BR", "BL"):
            x, y = fallback_corners[label]
            print(f"  {label} = ({x:.2f}, {y:.2f})")

    print()
    print("GRID / k (fase 6, a partir de las distancias entre markers ya rectificadas)")
    print("-----------------------------------------------------------------------------")
    if k_info is not None:
        print(f"  k = {k_info['k']} (estimado {k_info['k_float']:.3f})")
        print(f"  distancia horizontal (rectificada) = {k_info['horizontal_distance']:.2f}px")
        print(f"  distancia vertical   (rectificada) = {k_info['vertical_distance']:.2f}px")
        print(f"  tamaño de módulo estimado = {k_info['module_size']:.2f}px")
    else:
        print("  No se pudo deducir k.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) != 2:
        print(
            "Uso:\n"
            "  python3 lancherix_corner_rectifier_4markers.py imagen.png"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"ERROR: archivo no encontrado: {input_path}")
        sys.exit(1)

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"ERROR: no se pudo abrir: {input_path}")
        sys.exit(1)

    print()
    print("Lancherix 4-Marker Detector (con perspectiva)")
    print("================================================")
    print()
    print(f"Input: {input_path}")

    # ------------------------------------------------------------------------
    # FASE 1 — candidatos y cuadrilátero aproximado
    # ------------------------------------------------------------------------

    candidates = detect_white_center_candidates(image)
    best_quad = find_best_quad(candidates)
    marker_results = build_marker_results(best_quad)

    # ------------------------------------------------------------------------
    # FASE 2 — bandas internas (heredada)
    # ------------------------------------------------------------------------

    phase2_results = (
        analyze_four_markers(image, marker_results) if marker_results else None
    )

    # ------------------------------------------------------------------------
    # FASE 3/4 — lados reales (Hough) + cuadrilátero exacto
    # ------------------------------------------------------------------------

    marker_geoms = {}
    precise_result = None

    if marker_results:
        marker_geoms = detect_marker_geometries(image, marker_results)
        precise_result = build_precise_quad(marker_geoms)

    # ------------------------------------------------------------------------
    # FASE 5 — rectificación, con fallback si la fase 3/4 falló
    # ------------------------------------------------------------------------

    fallback_corners = (
        estimate_outer_corners_fallback(marker_results) if marker_results else None
    )

    rectified = None
    homography = None
    used_method = "none"

    if precise_result is not None:
        rectified, homography = rectify_from_corners(image, precise_result["corners"])
        if rectified is not None:
            used_method = "precise"

    if rectified is None and fallback_corners is not None:
        rectified, homography = rectify_from_corners(image, fallback_corners)
        if rectified is not None:
            used_method = "fallback"

    # ------------------------------------------------------------------------
    # FASE 6 (parte 1) — centros de marker rectificados, ANTES de corregir
    # orientación (la fase 5b los necesita en coordenadas del rectificado
    # crudo, para poder permutarlos junto con la imagen).
    # ------------------------------------------------------------------------

    rectified_centers = None

    if rectified is not None and homography is not None and marker_results:
        rectified_centers = rectify_marker_centers(marker_results, homography)

    # ------------------------------------------------------------------------
    # FASE 5b — corrección de orientación (rotación / espejo)
    #
    # Debe ejecutarse ANTES de calcular k y de leer la grilla de datos:
    # corrige rotated/mirrored el `rectified` y sus `rectified_centers`
    # para que todo lo que viene después (k, imagen de grilla, imagen
    # canónica, y el lector de símbolos) ya trabaje en la orientación
    # canónica del generador (TL=0°, TR=90°, BR=0°, BL=90°, sin rotar).
    # ------------------------------------------------------------------------

    orientation_correction = None

    if rectified is not None:
        orientation_correction = detect_orientation_correction(phase2_results)
        rectified, rectified_centers = apply_orientation_correction(
            rectified, rectified_centers, orientation_correction,
        )

        print()
        print("CORRECCION DE ORIENTACION (fase 5b)")
        print("--------------------------------------")
        print(f"  {orientation_correction['detail']}")
        print(
            f"  Aplicado: rotacion={orientation_correction['rotation']} "
            f"espejo={orientation_correction['mirror']}"
        )

    # ------------------------------------------------------------------------
    # FASE 6 (parte 2) — deducir k a partir de la cuadrícula YA corregida
    # ------------------------------------------------------------------------

    k_info = None
    if rectified_centers is not None:
        k_info = compute_k_and_module_size(rectified_centers)

    # ------------------------------------------------------------------------
    # REPORTE
    # ------------------------------------------------------------------------

    print_report(
        candidates, best_quad, marker_results, phase2_results,
        precise_result, fallback_corners, used_method, k_info,
    )

    # ------------------------------------------------------------------------
    # DEBUG
    # ------------------------------------------------------------------------

    overview_path = input_path.parent / f"{input_path.stem}_4markers_debug.png"
    overview = draw_overview_debug(
        image, candidates, marker_results or [], phase2_results, fallback_corners,
    )
    cv2.imwrite(str(overview_path), overview)
    print()
    print(f"Debug (overview) generado: {overview_path}")

    if marker_results:
        precise_path = input_path.parent / f"{input_path.stem}_4markers_precise.png"
        precise_debug = draw_precise_debug(image, marker_geoms, precise_result)
        cv2.imwrite(str(precise_path), precise_debug)
        print(f"Debug (líneas precisas) generado: {precise_path}")

    if rectified is not None:
        rectified_path = input_path.parent / f"{input_path.stem}_4markers_rectified.png"
        cv2.imwrite(str(rectified_path), rectified)
        print(f"Rectificada generada: {rectified_path}")
        print(
            f"Tamaño rectificado: {rectified.shape[1]}x{rectified.shape[0]} "
            f"(método: {used_method})"
        )

        if k_info is not None:
            grid_path = input_path.parent / f"{input_path.stem}_4markers_grid.png"
            grid_debug = draw_grid_debug(rectified, rectified_centers, k_info)
            cv2.imwrite(str(grid_path), grid_debug)
            print(f"Debug (cuadrícula, k={k_info['k']}) generado: {grid_path}")
    else:
        print("No se pudo generar la imagen rectificada.")


if __name__ == "__main__":
    main()