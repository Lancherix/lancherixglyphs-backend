"""
lancherix_generator.py
========================

Generador consolidado del Lancherix Visual Code V2 (con marcadores de
esquina y codec con ECC). Fusiona:

  - El generador de PRODUCCIÓN (antes lancherix_visual_v2_markers.py):
    render_png / render_svg / render, con colores y borde
    configurables por CLI.
  - El generador de imágenes de PRUEBA (antes _gen_test_image.py):
    modo normal (sin distorsión), modo perspectiva real (configurable:
    presets con nombre + parámetros finos + esquinas manuales), y modo
    afín (escala + rotación, sin perspectiva real).

CLI:
    python lancherix_generator.py real "texto o URL" salida
    python lancherix_generator.py test-normal "texto" salida.png
    python lancherix_generator.py test-perspective "texto" salida.png [opciones]
    python lancherix_generator.py test-affine "texto" salida.png

Ver `_build_arg_parser()` para el detalle completo de opciones.
"""

from __future__ import annotations
import sys

import cv2
import numpy as np
from PIL import Image

from lancherix_shapes import (
    QUIET_ZONE,
    MODULE_SIZE_PX,
    GRID_ASPECT_RATIO,
    END_GLYPH,
    DATA_GLYPHS,
    DEFAULT_FOREGROUND_COLOR,
    DEFAULT_BACKGROUND_COLOR,
    DEFAULT_BORDER_COLOR,
    BORDER_OUTER_MARGIN_MODULES,
    BORDER_THICKNESS_MODULES,
    BORDER_RADIUS_MODULES,
    DEFAULT_ECC_SYMBOLS,
    render_glyph,
    colorize_glyph,
    draw_rounded_border,
    border_svg_rects,
    glyph_svg_path_d,
    compute_grid_layout_with_markers,
    layout_glyph_rows_with_markers,
    rgb_to_hex,
    hex_to_rgb,
    encode_symbols,
)


# ==========================================================================
# CONSTRUCCIÓN DE LA GRILLA DE GLIFOS (compartida por real y test)
# ==========================================================================

def text_to_glyph_sequence(text: str, num_ecc_symbols: int = DEFAULT_ECC_SYMBOLS) -> list[tuple[str, int]]:
    """
    Convierte el texto en la secuencia plana de glifos de datos
    (SIN envolver en filas y SIN las 4 esquinas todavía): [d1, d2,
    ..., END]. Usa el codec CON corrección de errores.
    """
    values = encode_symbols(text, num_ecc_symbols=num_ecc_symbols)
    data_glyphs = [DATA_GLYPHS[v] for v in values]
    return data_glyphs + [END_GLYPH]


def build_rows(text: str, num_ecc_symbols: int = DEFAULT_ECC_SYMBOLS):
    """
    Construye la cuadrícula de filas de glifos (con las 4 esquinas
    reservadas) para `text`. Devuelve (rows, row_width, num_rows).
    """
    glyph_sequence = text_to_glyph_sequence(text, num_ecc_symbols=num_ecc_symbols)
    row_width, num_rows = compute_grid_layout_with_markers(len(glyph_sequence))
    rows = layout_glyph_rows_with_markers(glyph_sequence, row_width=row_width, num_rows=num_rows)
    return rows, row_width, num_rows


def build_code_image(text: str, num_ecc_symbols: int = DEFAULT_ECC_SYMBOLS,
                      module_size: int = MODULE_SIZE_PX) -> tuple[Image.Image, int, int]:
    """
    Genera la imagen del código SOLO (blanco/negro, sin borde), lista
    para pegar/distorsionar en un canvas de prueba más grande.
    Devuelve (imagen, row_width, num_rows).
    """
    rows, row_width, num_rows = build_rows(text, num_ecc_symbols=num_ecc_symbols)

    code_w = int(round((row_width + 2 * QUIET_ZONE) * module_size))
    code_h = int(round((num_rows + 2 * QUIET_ZONE) * module_size))

    image = Image.new("RGB", (code_w, code_h), DEFAULT_BACKGROUND_COLOR)

    for row_index, glyph_row in enumerate(rows):
        for col_index, glyph in enumerate(glyph_row):
            glyph_img = render_glyph(glyph[0], glyph[1], module_size)
            colored = colorize_glyph(glyph_img, DEFAULT_FOREGROUND_COLOR, DEFAULT_BACKGROUND_COLOR)
            x = int(round((QUIET_ZONE + col_index) * module_size))
            y = int(round((QUIET_ZONE + row_index) * module_size))
            image.paste(colored, (x, y))

    return image, row_width, num_rows


# ==========================================================================
# MODO REAL (producción): PNG + SVG con colores/borde configurables
# ==========================================================================

def render_png(text: str, output_path: str,
                module_size: int = MODULE_SIZE_PX,
                foreground_color=DEFAULT_FOREGROUND_COLOR,
                background_color=DEFAULT_BACKGROUND_COLOR,
                border_color=None,
                draw_border: bool = True,
                border_thickness_modules: float = BORDER_THICKNESS_MODULES,
                border_radius_modules: float = BORDER_RADIUS_MODULES,
                border_outer_margin_modules: float = BORDER_OUTER_MARGIN_MODULES,
                num_ecc_symbols: int = DEFAULT_ECC_SYMBOLS) -> None:
    """Genera el PNG de producción del Lancherix Visual Code para `text`."""
    if border_color is None:
        border_color = foreground_color

    rows, row_width, num_rows = build_rows(text, num_ecc_symbols=num_ecc_symbols)

    total_cols = 2 * QUIET_ZONE + row_width
    total_rows = 2 * QUIET_ZONE + num_rows

    img_width = round(total_cols * module_size)
    img_height = round(total_rows * module_size)

    canvas = Image.new("RGB", (img_width, img_height), background_color)

    if draw_border:
        draw_rounded_border(canvas, module_size, border_color, background_color,
                             outer_margin_modules=border_outer_margin_modules,
                             thickness_modules=border_thickness_modules,
                             radius_modules=border_radius_modules)

    for row_index, row in enumerate(rows):
        y = round((QUIET_ZONE + row_index) * module_size)
        for col_index, (family, orientation) in enumerate(row):
            gray_glyph = render_glyph(family, orientation, module_size)
            colored_glyph = colorize_glyph(gray_glyph, foreground_color, background_color)
            x = round((QUIET_ZONE + col_index) * module_size)
            canvas.paste(colored_glyph, (x, y))

    canvas.save(output_path, format="PNG")


def render_svg(text: str, output_path: str,
               module_size: float = MODULE_SIZE_PX,
               foreground_color=DEFAULT_FOREGROUND_COLOR,
               background_color=DEFAULT_BACKGROUND_COLOR,
               border_color=None,
               draw_border: bool = True,
               border_thickness_modules: float = BORDER_THICKNESS_MODULES,
               border_radius_modules: float = BORDER_RADIUS_MODULES,
               border_outer_margin_modules: float = BORDER_OUTER_MARGIN_MODULES,
               num_ecc_symbols: int = DEFAULT_ECC_SYMBOLS) -> None:
    """Genera el SVG de producción del Lancherix Visual Code para `text`."""
    if border_color is None:
        border_color = foreground_color

    rows, row_width, num_rows = build_rows(text, num_ecc_symbols=num_ecc_symbols)

    total_cols = 2 * QUIET_ZONE + row_width
    total_rows = 2 * QUIET_ZONE + num_rows

    width = total_cols * module_size
    height = total_rows * module_size

    bg_hex = rgb_to_hex(background_color)
    fg_hex = rgb_to_hex(foreground_color)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg_hex}" />',
    ]

    if draw_border:
        svg_parts.extend(
            border_svg_rects(width, height, module_size, border_color, background_color,
                              outer_margin_modules=border_outer_margin_modules,
                              thickness_modules=border_thickness_modules,
                              radius_modules=border_radius_modules)
        )

    for row_index, row in enumerate(rows):
        y = (QUIET_ZONE + row_index) * module_size
        for col_index, (family, orientation) in enumerate(row):
            if family == "empty":
                continue
            x = (QUIET_ZONE + col_index) * module_size
            path_d = glyph_svg_path_d(family, orientation, module_size)
            if path_d is None:
                continue
            svg_parts.append(
                f'<path d="{path_d}" fill="{fg_hex}" '
                f'transform="translate({x:.2f},{y:.2f})" />'
            )

    svg_parts.append("</svg>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))


def render(text: str, output_basepath: str,
           module_size: int = MODULE_SIZE_PX,
           foreground_color=DEFAULT_FOREGROUND_COLOR,
           background_color=DEFAULT_BACKGROUND_COLOR,
           border_color=None,
           draw_border: bool = True,
           border_thickness_modules: float = BORDER_THICKNESS_MODULES,
           border_radius_modules: float = BORDER_RADIUS_MODULES,
           border_outer_margin_modules: float = BORDER_OUTER_MARGIN_MODULES,
           num_ecc_symbols: int = DEFAULT_ECC_SYMBOLS) -> tuple[str, str]:
    """Genera tanto el PNG como el SVG, usando `output_basepath` como base."""
    base = output_basepath
    for ext in (".png", ".svg"):
        if base.endswith(ext):
            base = base[: -len(ext)]

    png_path = base + ".png"
    svg_path = base + ".svg"

    render_png(text, png_path, module_size=module_size,
               foreground_color=foreground_color, background_color=background_color,
               border_color=border_color, draw_border=draw_border,
               border_thickness_modules=border_thickness_modules,
               border_radius_modules=border_radius_modules,
               border_outer_margin_modules=border_outer_margin_modules,
               num_ecc_symbols=num_ecc_symbols)
    render_svg(text, svg_path, module_size=module_size,
               foreground_color=foreground_color, background_color=background_color,
               border_color=border_color, draw_border=draw_border,
               border_thickness_modules=border_thickness_modules,
               border_radius_modules=border_radius_modules,
               border_outer_margin_modules=border_outer_margin_modules,
               num_ecc_symbols=num_ecc_symbols)

    return png_path, svg_path


# ==========================================================================
# MODOS DE PRUEBA (test-normal / test-perspective / test-affine)
# ==========================================================================

_TEST_CANVAS_W = 1400
_TEST_CANVAS_H = 1000


def generate_normal_test_image(text: str, output_path: str,
                                canvas_w: int = _TEST_CANVAS_W,
                                canvas_h: int = _TEST_CANVAS_H,
                                num_ecc_symbols: int = DEFAULT_ECC_SYMBOLS) -> None:
    """
    Genera una imagen de prueba SIN ninguna transformación de
    perspectiva ni rotación: el código se pega "de frente" en el
    canvas, solo con un poco de escala para que no ocupe todo el
    canvas. Sirve como caso de control (sin distorsión).
    """
    code, row_width, num_rows = build_code_image(text, num_ecc_symbols=num_ecc_symbols)
    code_np = np.array(code)
    code_h, code_w = code_np.shape[:2]

    print(f"Código base: {code_w}x{code_h}px (row_width={row_width}, num_rows={num_rows}) -- texto={text!r}")

    scale = min((canvas_w * 0.7) / code_w, (canvas_h * 0.7) / code_h)
    scaled_w = int(round(code_w * scale))
    scaled_h = int(round(code_h * scale))

    scaled = cv2.resize(code_np, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    offset_x = (canvas_w - scaled_w) // 2
    offset_y = (canvas_h - scaled_h) // 2
    canvas[offset_y:offset_y + scaled_h, offset_x:offset_x + scaled_w] = scaled

    Image.fromarray(canvas).save(output_path)

    corners = np.array([
        [offset_x, offset_y],
        [offset_x + scaled_w, offset_y],
        [offset_x + scaled_w, offset_y + scaled_h],
        [offset_x, offset_y + scaled_h],
    ], dtype=np.float64)

    print()
    print("ESQUINAS REALES (sin perspectiva):")
    print(f"  TL = {tuple(corners[0])}")
    print(f"  TR = {tuple(corners[1])}")
    print(f"  BR = {tuple(corners[2])}")
    print(f"  BL = {tuple(corners[3])}")
    print()
    print(f"Generado: {output_path}")


# --------------------------------------------------------------------------
# Perspectiva configurable: preset con nombre (dirección de
# inclinación) + parámetro fino de intensidad ("strength"), o
# esquinas manuales (--corners) para control total.
# --------------------------------------------------------------------------

_TILT_DIRECTIONS = ("left", "right", "top", "bottom")


def compute_perspective_corners(canvas_w: int, canvas_h: int,
                                 tilt: str = "right", strength: float = 0.5,
                                 margin_frac: float = 0.2) -> np.ndarray:
    """
    Calcula las 4 esquinas destino (TL, TR, BR, BL) que simulan una
    foto tomada en ángulo, en función de:

      - `tilt`: qué lado del código queda "más lejos de la cámara"
        (más comprimido/inclinado): "left", "right", "top", "bottom".
      - `strength`: 0.0 (sin distorsión, rectángulo perfecto) a 1.0
        (inclinación extrema).

    Partiendo de un rectángulo base centrado en el canvas (definido
    por `margin_frac`), el lado elegido por `tilt` se desplaza hacia
    el centro y se inclina -- simulando que ese lado está más lejos.
    """
    if tilt not in _TILT_DIRECTIONS:
        raise ValueError(
            f"tilt inválido: {tilt!r}. Debe ser uno de {_TILT_DIRECTIONS}."
        )
    strength = max(0.0, min(1.0, float(strength)))

    w0 = canvas_w * (1 - 2 * margin_frac)
    h0 = canvas_h * (1 - 2 * margin_frac)
    x0 = canvas_w * margin_frac
    y0 = canvas_h * margin_frac

    tl = np.array([x0, y0], dtype=np.float64)
    tr = np.array([x0 + w0, y0], dtype=np.float64)
    br = np.array([x0 + w0, y0 + h0], dtype=np.float64)
    bl = np.array([x0, y0 + h0], dtype=np.float64)

    # Cuánto se desplaza cada esquina del lado "lejano": una fracción
    # del ancho/alto base, escalada por `strength`. El lado lejano se
    # acerca al centro horizontal/verticalmente (se "comprime") y
    # además se inclina (una punta más que la otra), para que el
    # resultado sea un trapecio con perspectiva real, no solo un
    # rectángulo más chico.
    compress = strength * 0.35   # cuánto se acerca el lado al centro
    skew = strength * 0.5        # cuánto se inclina ese mismo lado

    if tilt == "right":
        tr = tr + np.array([-w0 * compress, h0 * skew * 0.5])
        br = br + np.array([-w0 * compress, -h0 * skew * 0.5])
    elif tilt == "left":
        tl = tl + np.array([w0 * compress, h0 * skew * 0.5])
        bl = bl + np.array([w0 * compress, -h0 * skew * 0.5])
    elif tilt == "top":
        tl = tl + np.array([w0 * skew * 0.5, h0 * compress])
        tr = tr + np.array([-w0 * skew * 0.5, h0 * compress])
    elif tilt == "bottom":
        bl = bl + np.array([w0 * skew * 0.5, -h0 * compress])
        br = br + np.array([-w0 * skew * 0.5, -h0 * compress])

    return np.array([tl, tr, br, bl], dtype=np.float32)


def _parse_corners_arg(value: str) -> np.ndarray:
    """
    Parsea el argumento --corners: 4 pares "x,y" separados por
    espacios, en orden TL TR BR BL. Ej.:
        --corners "300,250 1100,150 1050,750 250,850"
    """
    parts = value.split()
    if len(parts) != 4:
        raise ValueError(
            "--corners requiere exactamente 4 puntos 'x,y' (TL TR BR BL), "
            f"separados por espacios; se recibieron {len(parts)}."
        )
    corners = []
    for part in parts:
        x_str, _, y_str = part.partition(",")
        corners.append([float(x_str), float(y_str)])
    return np.array(corners, dtype=np.float32)


def generate_perspective_test_image(text: str, output_path: str,
                                     dst_corners: np.ndarray | None = None,
                                     tilt: str = "right", strength: float = 0.5,
                                     canvas_w: int = _TEST_CANVAS_W,
                                     canvas_h: int = _TEST_CANVAS_H,
                                     num_ecc_symbols: int = DEFAULT_ECC_SYMBOLS) -> None:
    """
    Genera una imagen de prueba con perspectiva real (homografía) para
    `text`. Si `dst_corners` se indica explícitamente (4 puntos TL TR
    BR BL), se usa tal cual -- control total. Si no, las esquinas se
    calculan a partir de `tilt` + `strength` (ver
    `compute_perspective_corners`).
    """
    if dst_corners is None:
        dst_corners = compute_perspective_corners(canvas_w, canvas_h, tilt=tilt, strength=strength)

    code, row_width, num_rows = build_code_image(text, num_ecc_symbols=num_ecc_symbols)
    code_np = np.array(code)
    code_h, code_w = code_np.shape[:2]

    print(f"Código base: {code_w}x{code_h}px (row_width={row_width}, num_rows={num_rows}) -- texto={text!r}")

    src_corners = np.array([
        [0, 0], [code_w, 0], [code_w, code_h], [0, code_h],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_corners, dst_corners.astype(np.float32))

    warped = cv2.warpPerspective(
        code_np, matrix, (canvas_w, canvas_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )

    Image.fromarray(warped).save(output_path)

    print()
    print("ESQUINAS REALES (perspectiva):")
    print(f"  TL = {tuple(dst_corners[0])}")
    print(f"  TR = {tuple(dst_corners[1])}")
    print(f"  BR = {tuple(dst_corners[2])}")
    print(f"  BL = {tuple(dst_corners[3])}")
    print()
    print(f"Generado: {output_path}")


def generate_affine_test_image(text: str, output_path: str,
                                scale: float = 0.55, angle: float = 34.0,
                                canvas_w: int = _TEST_CANVAS_W,
                                canvas_h: int = _TEST_CANVAS_H,
                                num_ecc_symbols: int = DEFAULT_ECC_SYMBOLS) -> None:
    """
    Genera una imagen de prueba con escala + rotación (AFÍN, SIN
    perspectiva real). Modo opcional -- no es estrictamente necesario
    para el pipeline de cámara con corner markers (que ya cubre este
    caso como caso particular de perspectiva), pero se conserva para
    poder probar el caso más simple por separado.
    """
    code, row_width, num_rows = build_code_image(text, num_ecc_symbols=num_ecc_symbols)
    code_np = np.array(code)
    code_h, code_w = code_np.shape[:2]

    print(f"Código base: {code_w}x{code_h}px (row_width={row_width}, num_rows={num_rows}) -- texto={text!r}")

    scaled_w = int(round(code_w * scale))
    scaled_h = int(round(code_h * scale))
    scaled = cv2.resize(code_np, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    center = (scaled_w / 2.0, scaled_h / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    angle_rad = np.deg2rad(angle)
    cos_a = abs(np.cos(angle_rad))
    sin_a = abs(np.sin(angle_rad))
    rotated_w = int(np.ceil(scaled_w * cos_a + scaled_h * sin_a))
    rotated_h = int(np.ceil(scaled_w * sin_a + scaled_h * cos_a))

    rotation_matrix[0, 2] += rotated_w / 2.0 - center[0]
    rotation_matrix[1, 2] += rotated_h / 2.0 - center[1]

    rotated = cv2.warpAffine(
        scaled, rotation_matrix, (rotated_w, rotated_h),
        flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255),
    )

    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    offset_x = (canvas_w - rotated_w) // 2
    offset_y = (canvas_h - rotated_h) // 2
    canvas[offset_y:offset_y + rotated_h, offset_x:offset_x + rotated_w] = rotated

    Image.fromarray(canvas).save(output_path)

    corners = np.array([
        [0, 0, 1], [scaled_w, 0, 1], [scaled_w, scaled_h, 1], [0, scaled_h, 1],
    ], dtype=np.float64).T

    transformed_full = np.vstack([rotation_matrix.astype(np.float64), [0.0, 0.0, 1.0]])
    translation = np.array([
        [1.0, 0.0, offset_x], [0.0, 1.0, offset_y], [0.0, 0.0, 1.0],
    ])
    full_transform = translation @ transformed_full
    original_corners = (full_transform @ corners).T[:, :2]

    print()
    print("TRANSFORMACIÓN:")
    print(f"  escala = {scale}")
    print(f"  ángulo = {angle}°")
    print(f"  offset = ({offset_x}, {offset_y})")
    print()
    print("ESQUINAS REALES:")
    print(f"  TL = {tuple(original_corners[0])}")
    print(f"  TR = {tuple(original_corners[1])}")
    print(f"  BR = {tuple(original_corners[2])}")
    print(f"  BL = {tuple(original_corners[3])}")
    print()
    print(f"Generado: {output_path}")


# ==========================================================================
# CLI
# ==========================================================================

def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generador del Lancherix Visual Code V2 (producción y pruebas).",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- real ---
    p_real = subparsers.add_parser("real", help="Generar el código de producción (PNG+SVG).")
    p_real.add_argument("texto", help="Texto o URL a codificar")
    p_real.add_argument("salida", nargs="?", default="lancherix_visual",
                         help="Nombre base de salida (sin extensión).")
    p_real.add_argument("--fg", "--foreground", dest="fg", default=None,
                         help="Color de las formas/código, en hex. Por defecto: negro.")
    p_real.add_argument("--bg", "--background", dest="bg", default=None,
                         help="Color de fondo, en hex. Por defecto: blanco.")
    p_real.add_argument("--border-color", dest="border_color", default=None,
                         help="Color del borde, en hex. Por defecto: igual a --fg.")
    p_real.add_argument("--no-border", dest="no_border", action="store_true",
                         help="No dibujar el borde redondeado.")
    p_real.add_argument("--border-thickness", dest="border_thickness", type=float, default=None)
    p_real.add_argument("--border-radius", dest="border_radius", type=float, default=None)
    p_real.add_argument("--border-margin", dest="border_margin", type=float, default=None)
    p_real.add_argument("--ecc", dest="ecc", type=int, default=None,
                         help=f"Símbolos de paridad Reed-Solomon (bytes). Por defecto: {DEFAULT_ECC_SYMBOLS}.")

    # --- test-normal ---
    p_normal = subparsers.add_parser("test-normal", help="Imagen de prueba sin distorsión.")
    p_normal.add_argument("texto")
    p_normal.add_argument("salida", nargs="?", default="test_normal.png")
    p_normal.add_argument("--ecc", type=int, default=None)

    # --- test-perspective ---
    p_persp = subparsers.add_parser("test-perspective", help="Imagen de prueba con perspectiva real, configurable.")
    p_persp.add_argument("texto")
    p_persp.add_argument("salida", nargs="?", default="test_perspective.png")
    p_persp.add_argument("--tilt", choices=_TILT_DIRECTIONS, default="right",
                          help="Qué lado del código queda 'más lejos' de la cámara. Por defecto: right.")
    p_persp.add_argument("--strength", type=float, default=0.5,
                          help="Intensidad de la inclinación, de 0.0 (nada) a 1.0 (extrema). Por defecto: 0.5.")
    p_persp.add_argument("--corners", dest="corners", default=None,
                          help="4 esquinas manuales 'x,y' (TL TR BR BL) separadas por espacios; "
                               "si se indica, ignora --tilt/--strength. Ej.: "
                               "\"300,250 1100,150 1050,750 250,850\"")
    p_persp.add_argument("--canvas-width", type=int, default=_TEST_CANVAS_W)
    p_persp.add_argument("--canvas-height", type=int, default=_TEST_CANVAS_H)
    p_persp.add_argument("--ecc", type=int, default=None)

    # --- test-affine ---
    p_affine = subparsers.add_parser("test-affine", help="Imagen de prueba con escala+rotación (sin perspectiva real).")
    p_affine.add_argument("texto")
    p_affine.add_argument("salida", nargs="?", default="test_affine.png")
    p_affine.add_argument("--scale", type=float, default=0.55)
    p_affine.add_argument("--angle", type=float, default=34.0)
    p_affine.add_argument("--ecc", type=int, default=None)

    return parser


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "real":
        foreground_color = hex_to_rgb(args.fg) if args.fg else DEFAULT_FOREGROUND_COLOR
        background_color = hex_to_rgb(args.bg) if args.bg else DEFAULT_BACKGROUND_COLOR
        border_color = hex_to_rgb(args.border_color) if args.border_color else None

        kwargs = dict(
            foreground_color=foreground_color,
            background_color=background_color,
            border_color=border_color,
            draw_border=not args.no_border,
        )
        if args.border_thickness is not None:
            kwargs["border_thickness_modules"] = args.border_thickness
        if args.border_radius is not None:
            kwargs["border_radius_modules"] = args.border_radius
        if args.border_margin is not None:
            kwargs["border_outer_margin_modules"] = args.border_margin
        if args.ecc is not None:
            kwargs["num_ecc_symbols"] = args.ecc

        png_path, svg_path = render(args.texto, args.salida, **kwargs)
        print(f"Generado: {png_path}")
        print(f"Generado: {svg_path}")
        print(f"Texto codificado: {args.texto}")

    elif args.mode == "test-normal":
        kwargs = {}
        if args.ecc is not None:
            kwargs["num_ecc_symbols"] = args.ecc
        generate_normal_test_image(args.texto, args.salida, **kwargs)

    elif args.mode == "test-perspective":
        kwargs = dict(canvas_w=args.canvas_width, canvas_h=args.canvas_height)
        if args.ecc is not None:
            kwargs["num_ecc_symbols"] = args.ecc
        if args.corners is not None:
            dst_corners = _parse_corners_arg(args.corners)
            generate_perspective_test_image(args.texto, args.salida, dst_corners=dst_corners, **kwargs)
        else:
            generate_perspective_test_image(
                args.texto, args.salida, tilt=args.tilt, strength=args.strength, **kwargs)

    elif args.mode == "test-affine":
        kwargs = {}
        if args.ecc is not None:
            kwargs["num_ecc_symbols"] = args.ecc
        generate_affine_test_image(args.texto, args.salida, scale=args.scale, angle=args.angle, **kwargs)


if __name__ == "__main__":
    main()