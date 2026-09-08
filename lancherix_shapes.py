"""
lancherix_shapes.py
====================

Módulo base del Lancherix Visual Code, consolidado. Contiene:

  1. Geometría de los glifos (formas + rotaciones), grilla adaptativa
     3:1 CON marcadores de esquina, borde redondeado y utilidades de
     color.
  2. El codec V2 completo CON corrección de errores (header, CRC-8,
     Reed-Solomon sobre GF(256) vía `reedsolo`, interleaving a nivel
     de bloque). Requiere `pip install reedsolo`.
  3. Utilidades de detección de color y template-matching (rescatadas
     y generalizadas de la versión anterior de lancherix_reader.py),
     usadas por el pipeline de cámara para resolver fondo/forma y
     leer cada celda de la grilla.

--------------------------------------------------------------------------
IDEA DEL DISEÑO (formas)
--------------------------------------------------------------------------

Cada módulo de la grilla es una FORMA con una ROTACIÓN:

    - Cuarto de círculo, en 4 rotaciones (0°, 90°, 180°, 270°)  -> 4 estados
    - Semicírculo,       en 4 rotaciones (0°, 90°, 180°, 270°)  -> 4 estados

    8 estados en total = 3 bits por módulo (log2(8) = 3)

Además, 3 de las 4 esquinas de la grilla llevan un glifo especial
"corner_marker" (medio cuadrado + medio círculo, SIN ninguna simetría
de rotación: sus 4 orientaciones son todas mutuamente distintas), que
permite resolver sin ambigüedad la orientación real de una foto con
perspectiva -- igual que los finder patterns de un QR.

Asignación esquina -> orientación:
    arriba-izquierda (TL) -> CORNER_MARKER_GLYPHS[0] (0°,  bombeo izquierda)
    arriba-derecha   (TR) -> CORNER_MARKER_GLYPHS[1] (90°, bombeo arriba)
    abajo-izquierda  (BL) -> CORNER_MARKER_GLYPHS[3] (270°, bombeo abajo)
    abajo-derecha    (BR) -> sin marcador (glifo de datos normal)

--------------------------------------------------------------------------
CUADRÍCULA ADAPTATIVA CON RELACIÓN 3:1 (columnas : filas)
--------------------------------------------------------------------------

La cuadrícula de datos se adapta al texto manteniendo siempre una
relación 3:1 entre columnas y filas:

    k    ->  k filas x (3k) columnas (capacidad: 3k² módulos, menos
             3 celdas reservadas para los corner markers)

--------------------------------------------------------------------------
CODEC V2 CON ECC
--------------------------------------------------------------------------

El texto se codifica en bytes UTF-8 directos (no Base64):

    [header(version 1 byte + payload_length varint)] + [payload]
    + [checksum CRC-8] + [paridad Reed-Solomon (num_ecc_symbols bytes)]

seguido de interleaving a nivel de bloque RS (no-op si el mensaje
cabe en un solo bloque), y finalmente partido en símbolos de 3 bits.

`DEFAULT_ECC_SYMBOLS = 4` corrige hasta ~2 bytes erróneos por bloque
de 255 bytes. El mismo valor debe usarse al codificar y al decodificar.

--------------------------------------------------------------------------
FLUJO COMPLETO (encoder / decoder)
--------------------------------------------------------------------------

Codificar:
    values = encode_symbols(text, num_ecc_symbols)
    glyph_sequence = [DATA_GLYPHS[v] for v in values] + [END_GLYPH]
    -> layout_glyph_rows_with_markers(...) (rellena con datos
       decorativos y reserva las 3 esquinas)

Decodificar:
    leer la grilla completa -> excluir las 3 celdas de esquina (por
    índice: 0, row_width-1, (num_rows-1)*row_width) -> buscar el
    índice de END_GLYPH -> tomar todo lo anterior -> mapear cada
    glifo a su valor con GLYPH_TO_VALUE -> decode_symbols(values,
    num_ecc_symbols)
"""

from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw

from reedsolo import RSCodec, ReedSolomonError

# --------------------------------------------------------------------------
# Configuración del diseño
# --------------------------------------------------------------------------

QUIET_ZONE = 0.5     # 0.3 Grosor del margen (en módulos), a cada lado
MODULE_SIZE_PX = 40      # Tamaño en píxeles de cada módulo cuadrado

# Relación fija columnas:filas del área de datos (ancho:alto = 3:1).
GRID_ASPECT_RATIO = 3

BITS_PER_MODULE = 3      # cuarto/semicírculo x 4 rotaciones = 8 = 2**3

# Colores por defecto: formas negras sobre fondo blanco, borde negro.
DEFAULT_FOREGROUND_COLOR = (0, 0, 0)
DEFAULT_BACKGROUND_COLOR = (255, 255, 255)
DEFAULT_BORDER_COLOR = (0, 0, 0)

# --------------------------------------------------------------------------
# Geometría del borde redondeado (en unidades de módulo)
# --------------------------------------------------------------------------

BORDER_OUTER_MARGIN_MODULES = 0   # separación entre el borde exterior y el borde de la imagen
BORDER_THICKNESS_MODULES = 0.12      # grosor del trazo del borde
BORDER_RADIUS_MODULES = 0.6         # radio de las esquinas redondeadas

# Las 8 formas de datos posibles, en el orden que define su valor 0-7.
# (familia, rotación_en_grados)
DATA_GLYPHS = [
    ("quarter", 0), ("quarter", 90), ("quarter", 180), ("quarter", 270),
    ("half", 0), ("half", 90), ("half", 180), ("half", 270),
]
GLYPH_TO_VALUE = {glyph: value for value, glyph in enumerate(DATA_GLYPHS)}

# Marcador de fin de datos reales (la "hoja"/vesica, orientación 90°).
# START_GLYPH ya no se usa: su rol de orientación lo cumplen los 3
# corner_marker (vía homografía).
END_GLYPH = ("leaf", 90)      # diagonal ↙

EMPTY_GLYPH = ("empty", 0)    # módulo en blanco (zona de silencio)

# Marcadores de esquina (medio cuadrado + medio círculo, sin simetría
# de rotación -- las 4 orientaciones son todas mutuamente distintas).
CORNER_MARKER_GLYPHS = [
    ("corner_marker", 0),     # bombeo hacia la izquierda (rectángulo a la derecha)
    ("corner_marker", 90),    # bombeo hacia arriba      (rectángulo abajo)
    ("corner_marker", 180),   # bombeo hacia la derecha   (rectángulo a la izquierda)
    ("corner_marker", 270),   # bombeo hacia abajo        (rectángulo arriba)
]

# Todos los glifos posibles (usado por el lector para "template
# matching"). Incluye los corner markers para que el matching sea
# correcto también en esas 3 celdas -- aunque el decodificador las
# descarta por ÍNDICE (no por contenido), un match incorrecto ahí
# metería error de más en el score total usado para resolver la
# polaridad fondo/forma.
ALL_GLYPHS = [EMPTY_GLYPH, END_GLYPH] + DATA_GLYPHS + CORNER_MARKER_GLYPHS

# --------------------------------------------------------------------------
# Dibujo de un glifo individual
# --------------------------------------------------------------------------

def render_glyph(family: str, orientation: int, size: int) -> Image.Image:
    """
    Dibuja un solo glifo (forma + rotación) en una imagen cuadrada de
    `size` x `size` píxeles, en escala de grises ('L'): fondo blanco
    (255), forma en negro (0).
    """
    img = Image.new("L", (size, size), 255)

    if family == "empty":
        return img

    draw = ImageDraw.Draw(img)

    if family == "quarter":
        corner_by_orientation = {
            0: (0, 0, 0, 90),
            90: (size, 0, 90, 180),
            180: (size, size, 180, 270),
            270: (0, size, 270, 360),
        }
        cx, cy, angle_start, angle_end = corner_by_orientation[orientation]
        bbox = [cx - size, cy - size, cx + size, cy + size]
        draw.pieslice(bbox, angle_start, angle_end, fill=0)

    elif family == "half":
        r = size / 2
        edge_by_orientation = {
            0: (size / 2, 0, 0, 180),
            90: (size, size / 2, 90, 270),
            180: (size / 2, size, 180, 360),
            270: (0, size / 2, 270, 450),
        }
        cx, cy, angle_start, angle_end = edge_by_orientation[orientation]
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.pieslice(bbox, angle_start, angle_end, fill=0)

    elif family == "leaf":
        if orientation == 0:
            c1, c2 = (0, 0), (size, size)      # diagonal ↘
        else:
            c1, c2 = (size, 0), (0, size)      # diagonal ↙

        r = size
        mask1 = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask1).ellipse(
            [c1[0] - r, c1[1] - r, c1[0] + r, c1[1] + r], fill=255)
        mask2 = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask2).ellipse(
            [c2[0] - r, c2[1] - r, c2[0] + r, c2[1] + r], fill=255)

        arr1 = np.array(mask1) > 128
        arr2 = np.array(mask2) > 128
        inter = arr1 & arr2

        out = np.full((size, size), 255, dtype=np.uint8)
        out[inter] = 0
        img = Image.fromarray(out, mode="L")

    elif family == "corner_marker":
        r = size / 2
        cx, cy = size / 2, size / 2
        rect_by_orientation = {
            0: [cx, 0, size, size],    # bombeo izquierda -> rectángulo a la derecha
            90: [0, cy, size, size],   # bombeo arriba    -> rectángulo abajo
            180: [0, 0, cx, size],     # bombeo derecha    -> rectángulo a la izquierda
            270: [0, 0, size, cy],     # bombeo abajo      -> rectángulo arriba
        }
        angle_by_orientation = {
            0: (90, 270),
            90: (180, 360),
            180: (270, 450),
            270: (0, 180),
        }

        draw.rectangle(rect_by_orientation[orientation], fill=0)
        angle_start, angle_end = angle_by_orientation[orientation]
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.pieslice(bbox, angle_start, angle_end, fill=0)

        # Punto central blanco (radio = 12% del módulo).
        center_dot_radius = size * 0.12
        dot_bbox = [
            cx - center_dot_radius, cy - center_dot_radius,
            cx + center_dot_radius, cy + center_dot_radius,
        ]
        draw.ellipse(dot_bbox, fill=255)

    else:
        raise ValueError(f"Familia de glifo desconocida: {family}")

    return img


def colorize_glyph(gray_img: Image.Image, foreground_color, background_color) -> Image.Image:
    """
    Convierte un glifo en escala de grises (0=negro/forma,
    255=blanco/fondo) a una imagen RGB usando los colores elegidos,
    preservando el antialiasing.
    """
    gray = np.asarray(gray_img, dtype=np.float64) / 255.0  # 1.0=fondo, 0.0=forma
    fg = np.asarray(foreground_color, dtype=np.float64)
    bg = np.asarray(background_color, dtype=np.float64)
    rgb = gray[..., None] * bg + (1.0 - gray[..., None]) * fg
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def draw_rounded_border(canvas: Image.Image, module_size: float,
                         border_color, background_color,
                         outer_margin_modules: float = BORDER_OUTER_MARGIN_MODULES,
                         thickness_modules: float = BORDER_THICKNESS_MODULES,
                         radius_modules: float = BORDER_RADIUS_MODULES) -> None:
    """
    Dibuja, DENTRO de la zona de silencio, un borde rectangular con
    esquinas redondeadas alrededor de todo el código.
    """
    width, height = canvas.size
    margin = outer_margin_modules * module_size
    thickness = thickness_modules * module_size
    radius_outer = radius_modules * module_size

    draw = ImageDraw.Draw(canvas)

    outer_box = [margin, margin, width - margin, height - margin]
    draw.rounded_rectangle(outer_box, radius=radius_outer, fill=border_color)

    inner_margin = margin + thickness
    radius_inner = max(radius_outer - thickness, 0)
    inner_box = [inner_margin, inner_margin, width - inner_margin, height - inner_margin]
    draw.rounded_rectangle(inner_box, radius=radius_inner, fill=background_color)


def border_svg_rects(width: float, height: float, module_size: float,
                      border_color, background_color,
                      outer_margin_modules: float = BORDER_OUTER_MARGIN_MODULES,
                      thickness_modules: float = BORDER_THICKNESS_MODULES,
                      radius_modules: float = BORDER_RADIUS_MODULES) -> list[str]:
    """
    Devuelve las dos etiquetas <rect> (borde exterior + relleno
    interior) que reproducen, en SVG, el mismo borde que
    `draw_rounded_border` dibuja en el PNG.
    """
    margin = outer_margin_modules * module_size
    thickness = thickness_modules * module_size
    radius_outer = radius_modules * module_size
    inner_margin = margin + thickness
    radius_inner = max(radius_outer - thickness, 0)

    border_hex = rgb_to_hex(border_color)
    bg_hex = rgb_to_hex(background_color)

    return [
        f'<rect x="{margin:.2f}" y="{margin:.2f}" '
        f'width="{width - 2 * margin:.2f}" height="{height - 2 * margin:.2f}" '
        f'rx="{radius_outer:.2f}" fill="{border_hex}" />',
        f'<rect x="{inner_margin:.2f}" y="{inner_margin:.2f}" '
        f'width="{width - 2 * inner_margin:.2f}" height="{height - 2 * inner_margin:.2f}" '
        f'rx="{radius_inner:.2f}" fill="{bg_hex}" />',
    ]


def rgb_to_hex(color) -> str:
    r, g, b = (int(c) for c in color[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """
    Convierte un color en formato hexadecimal ('#rrggbb', '#rgb', o
    sin el '#') a una tupla (r, g, b) de enteros 0-255.
    """
    color = color.strip().lstrip("#")
    if len(color) == 3:
        color = "".join(ch * 2 for ch in color)
    if len(color) != 6:
        raise ValueError(f"Color hexadecimal inválido: {color!r}")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


# Alias retrocompatible.
_rgb_to_hex = rgb_to_hex


# --------------------------------------------------------------------------
# Grilla adaptativa CON marcadores de esquina
# --------------------------------------------------------------------------

def compute_grid_layout_with_markers(glyph_sequence_length: int) -> tuple[int, int]:
    """
    Elige el k más chico tal que una cuadrícula de k filas x 3k
    columnas (relación 3:1) tenga capacidad suficiente para
    `glyph_sequence_length` glifos de datos (SIN contar las 3
    esquinas reservadas, que se restan de la capacidad: 3k² - 3).
    Exige k >= 2 (con k=1 no existen 3 esquinas distintas).
    """
    k = 2
    while GRID_ASPECT_RATIO * k * k - 4 < glyph_sequence_length:
        k += 1
    return GRID_ASPECT_RATIO * k, k


def layout_glyph_rows_with_markers(
    glyph_sequence: list[tuple[str, int]],
    row_width: int,
    num_rows: int | None = None,
) -> list[list[tuple[str, int]]]:
    """
    Distribuye `glyph_sequence` (datos + END + relleno decorativo, SIN
    START) en una cuadrícula de `num_rows` x `row_width`, rreservando las 4 esquinas (arriba-izquierda, arriba-derecha,
    abajo-derecha, abajo-izquierda) para `corner_marker`. Si sobran celdas se rellenan con
    datos decorativos cíclicos para que la figura sea un rectángulo
    sólido sin huecos.
    """
    if row_width < 2 or (num_rows is not None and num_rows < 2):
        raise ValueError(
            "Se requieren al menos 2 columnas y 2 filas para tener 4 "
            "esquinas distintas."
        )

    if num_rows is None:
        num_rows = 2
        while row_width * num_rows - 4 < len(glyph_sequence):
            num_rows += 1

    if num_rows < 2:
        raise ValueError(
            "Se requieren al menos 2 filas para tener 3 esquinas distintas."
        )

    total_cells = row_width * num_rows
    capacity = total_cells - 4

    corner_glyphs = {
        (0, 0): CORNER_MARKER_GLYPHS[0],                        # TL = 0°
        (0, row_width - 1): CORNER_MARKER_GLYPHS[1],            # TR = 90°
        (num_rows - 1, row_width - 1): CORNER_MARKER_GLYPHS[0], # BR = 0° (igual a TL)
        (num_rows - 1, 0): CORNER_MARKER_GLYPHS[1],              # BL = 90° (igual a TR)
    }
    if len(corner_glyphs) != 4:
        raise ValueError(
            "La grilla es demasiado chica para tener 4 esquinas distintas."
        )

    data = list(glyph_sequence)
    if len(data) > capacity:
        raise ValueError(
            f"La secuencia ({len(data)} glifos) no cabe en la grilla "
            f"({capacity} celdas disponibles tras reservar 4 esquinas)."
        )
    if len(data) < capacity:
        pad_count = capacity - len(data)
        filler = [
            DATA_GLYPHS[(len(data) + j) % len(DATA_GLYPHS)]
            for j in range(pad_count)
        ]
        data = data + filler

    data_iter = iter(data)
    rows: list[list[tuple[str, int]]] = []
    for r in range(num_rows):
        row = []
        for c in range(row_width):
            if (r, c) in corner_glyphs:
                row.append(corner_glyphs[(r, c)])
            else:
                row.append(next(data_iter))
        rows.append(row)

    return rows


# --------------------------------------------------------------------------
# Contorno vectorial de un glifo (para exportar a SVG)
# --------------------------------------------------------------------------

_CONTOUR_RESOLUTION = 300


def glyph_svg_path_d(family: str, orientation: int, size: float) -> str | None:
    """
    Devuelve el atributo `d` de un <path> SVG que dibuja exactamente
    la misma forma que produce render_glyph(), escalada a `size`.

    Incluye TODOS los contornos de la forma (no solo el más largo)
    como subpaths separados dentro del mismo `d` -- esto es necesario
    para glifos con agujeros internos, como el punto central blanco
    del `corner_marker`. El <path> resultante debe dibujarse con
    fill-rule="evenodd" para que esos agujeros se rendericen
    correctamente como huecos (del color de fondo) en vez de quedar
    rellenos.
    """
    from skimage import measure  # import perezoso: solo se necesita para SVG

    res = _CONTOUR_RESOLUTION
    mask_img = render_glyph(family, orientation, res)
    arr = np.array(mask_img) < 128  # True donde es negro (la forma)

    if not arr.any():
        return None

    padded = np.pad(arr, pad_width=1, mode="constant", constant_values=False)

    contours = measure.find_contours(padded.astype(float), level=0.5)
    if not contours:
        return None

    scale = size / res
    subpaths = []
    for contour in contours:
        points = [((x - 1) * scale, (y - 1) * scale) for y, x in contour]
        coords = " L ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        subpaths.append(f"M {coords} Z")

    return " ".join(subpaths)


# ==========================================================================
# CODEC V2 CON CORRECCIÓN DE ERRORES (fusionado de lancherix_v2_codec.py)
# ==========================================================================

# Versión del formato V2.
FORMAT_VERSION = 2

# Símbolos de paridad Reed-Solomon (en BYTES) por defecto.
DEFAULT_ECC_SYMBOLS = 4

_BITS_PER_SYMBOL = 3
_MAX_SYMBOL_VALUE = (1 << _BITS_PER_SYMBOL) - 1  # 7


# --------------------------------------------------------------------------
# Varint (estilo Protobuf / UTF-8): 7 bits de valor + 1 bit de
# continuación (MSB) por byte. Sin límite superior de tamaño.
# --------------------------------------------------------------------------

def _encode_varint(n: int) -> bytes:
    if n < 0:
        raise ValueError("payload_length no puede ser negativo.")
    out = bytearray()
    while True:
        chunk = n & 0x7F
        n >>= 7
        if n:
            out.append(chunk | 0x80)
        else:
            out.append(chunk)
            return bytes(out)


def _decode_varint_from_bytes(data: bytes, pos: int) -> tuple[int, int]:
    """
    Lee un varint a partir de `pos` en `data` (bytes ya alineados,
    después de la corrección ECC). Devuelve (valor, nueva_posición).
    """
    value = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise ValueError(
                "Secuencia incompleta: no se pudo leer el campo "
                "payload_length (varint) del header."
            )
        byte_val = data[pos]
        pos += 1
        value |= (byte_val & 0x7F) << shift
        shift += 7
        if not (byte_val & 0x80):
            return value, pos


# --------------------------------------------------------------------------
# CRC-8 (detección de corrupción -- NO corrección; eso lo hace el ECC)
# --------------------------------------------------------------------------

_CRC8_POLY = 0x07   # variante CRC-8/SMBUS: polinomio 0x07, sin reflejar
_CRC8_INIT = 0x00


def _crc8(data: bytes) -> int:
    crc = _CRC8_INIT
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ _CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


# --------------------------------------------------------------------------
# Conversión bytes <-> bits <-> símbolos de 3 bits
# --------------------------------------------------------------------------

def _bytes_to_bits(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)


def _bits_to_bytes(bits: str) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("La cantidad de bits no es múltiplo de 8.")
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def _bits_to_symbols(bits: str) -> list[int]:
    pad = (-len(bits)) % _BITS_PER_SYMBOL
    bits = bits + "0" * pad
    return [int(bits[i:i + _BITS_PER_SYMBOL], 2)
            for i in range(0, len(bits), _BITS_PER_SYMBOL)]


def _symbols_to_bits(symbols: list[int]) -> str:
    for s in symbols:
        if not (0 <= s <= _MAX_SYMBOL_VALUE):
            raise ValueError(f"Símbolo fuera de rango (0-{_MAX_SYMBOL_VALUE}): {s!r}")
    return "".join(f"{s:0{_BITS_PER_SYMBOL}b}" for s in symbols)


# --------------------------------------------------------------------------
# Interleaving (a nivel de BLOQUE Reed-Solomon, en bytes)
# --------------------------------------------------------------------------
#
# Solo se aplica cuando el mensaje ocupa más de un bloque RS (a partir
# de 255 bytes de datos); en el caso común (un solo bloque) es un
# no-op, ver nota de diseño en la documentación original del codec.

_RS_BLOCK_SIZE = 255  # nsize por defecto de `reedsolo` (GF(256))


def _rs_block_lengths(total_encoded_len: int, num_ecc_symbols: int,
                       block_size: int = _RS_BLOCK_SIZE) -> list[int]:
    """
    Reconstruye el largo (en bytes) de cada bloque codificado por
    Reed-Solomon, de forma determinística a partir del largo total.
    """
    if total_encoded_len <= 0:
        return []
    full_blocks, remainder = divmod(total_encoded_len, block_size)
    lengths = [block_size] * full_blocks
    if remainder:
        lengths.append(remainder)
    return lengths


def interleave_bytes(data: bytes, num_ecc_symbols: int) -> bytes:
    """
    Entrelaza `data` a nivel de BLOQUE: si hay más de un bloque,
    escribe cada bloque como una "fila" y lee columna por columna.
    No-op si el mensaje cabe en un único bloque RS.
    """
    block_lengths = _rs_block_lengths(len(data), num_ecc_symbols)
    if len(block_lengths) <= 1:
        return data

    offsets = [0]
    for length in block_lengths:
        offsets.append(offsets[-1] + length)

    max_len = max(block_lengths)
    out = bytearray()
    for col in range(max_len):
        for block_idx, length in enumerate(block_lengths):
            if col < length:
                out.append(data[offsets[block_idx] + col])
    return bytes(out)


def deinterleave_bytes(data: bytes, num_ecc_symbols: int) -> bytes:
    """
    Proceso inverso exacto de interleave_bytes().
    """
    block_lengths = _rs_block_lengths(len(data), num_ecc_symbols)
    if len(block_lengths) <= 1:
        return data

    offsets = [0]
    for length in block_lengths:
        offsets.append(offsets[-1] + length)

    max_len = max(block_lengths)
    out = bytearray(len(data))
    pos = 0
    for col in range(max_len):
        for block_idx, length in enumerate(block_lengths):
            if col < length:
                out[offsets[block_idx] + col] = data[pos]
                pos += 1
    return bytes(out)


# --------------------------------------------------------------------------
# API pública del codec: texto <-> símbolos de 3 bits
# --------------------------------------------------------------------------

def encode_symbols(text: str, num_ecc_symbols: int = DEFAULT_ECC_SYMBOLS) -> list[int]:
    """
    Codifica `text` (UTF-8 arbitrario) como una lista de símbolos de
    3 bits (valores 0-7):

        [header(version + varint length)] + [payload] + [checksum CRC-8]
        + [paridad Reed-Solomon (num_ecc_symbols bytes)]

    `num_ecc_symbols` debe ser el MISMO valor usado luego en
    `decode_symbols`.
    """
    payload = text.encode("utf-8")
    header = bytes([FORMAT_VERSION]) + _encode_varint(len(payload))
    header_and_payload = header + payload
    checksum = _crc8(header_and_payload)
    data = header_and_payload + bytes([checksum])

    rsc = RSCodec(num_ecc_symbols)
    data_with_ecc = bytes(rsc.encode(data))
    data_with_ecc = interleave_bytes(data_with_ecc, num_ecc_symbols)

    bits = _bytes_to_bits(data_with_ecc)
    return _bits_to_symbols(bits)


def decode_symbols(symbols: list[int], num_ecc_symbols: int = DEFAULT_ECC_SYMBOLS) -> str:
    """
    Proceso inverso de encode_symbols(): reconstruye el texto original
    a partir de la lista de símbolos de 3 bits, corrigiendo errores
    con Reed-Solomon cuando es posible.

    Lanza ValueError si hay más errores de los que el ECC puede
    corregir, la versión no es soportada, la secuencia está
    incompleta, o el checksum no coincide incluso después de la
    corrección ECC.
    """
    bits = _symbols_to_bits(symbols)

    total_bytes = len(bits) // 8
    if total_bytes <= num_ecc_symbols:
        raise ValueError(
            "Secuencia demasiado corta para contener siquiera la "
            "paridad ECC esperada."
        )

    data_with_ecc = _bits_to_bytes(bits[:total_bytes * 8])
    data_with_ecc = deinterleave_bytes(data_with_ecc, num_ecc_symbols)

    rsc = RSCodec(num_ecc_symbols)
    try:
        decoded, _decoded_with_ecc, _errata_pos = rsc.decode(data_with_ecc)
    except ReedSolomonError as exc:
        raise ValueError(
            "No se pudo corregir el código: hay demasiados errores "
            f"para el nivel de ECC configurado ({num_ecc_symbols} "
            f"símbolos de paridad). Detalle: {exc}"
        )
    decoded = bytes(decoded)

    if len(decoded) < 2:
        raise ValueError("Secuencia demasiado corta: falta el header.")

    version = decoded[0]
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Versión de formato no soportada: {version} "
            f"(se esperaba {FORMAT_VERSION})."
        )

    payload_length, pos = _decode_varint_from_bytes(decoded, 1)

    payload_bytes = decoded[pos:pos + payload_length]
    if len(payload_bytes) < payload_length:
        raise ValueError(
            "Secuencia incompleta: faltan bytes del payload "
            f"(se esperaban {payload_length} bytes)."
        )
    pos += payload_length

    if pos >= len(decoded):
        raise ValueError("Secuencia incompleta: falta el byte de checksum.")
    stored_checksum = decoded[pos]
    computed_checksum = _crc8(decoded[:pos])

    if stored_checksum != computed_checksum:
        raise ValueError(
            "Checksum inválido incluso después de la corrección ECC: "
            "la reconstrucción no es confiable (esperado "
            f"0x{computed_checksum:02x}, encontrado 0x{stored_checksum:02x})."
        )

    return payload_bytes.decode("utf-8")


# ==========================================================================
# UTILIDADES DE DETECCIÓN DE COLOR Y TEMPLATE-MATCHING
# (rescatadas y generalizadas de la versión anterior de lancherix_reader.py)
# ==========================================================================

# Tamaño (px) al que se normaliza cada módulo recortado antes de
# compararlo con las plantillas, y al que se renderizan las
# plantillas mismas por defecto. Parametrizable: distintos módulos
# (ej. el warp canónico de la cámara) pueden pasar otro valor.
_TEMPLATE_SIZE = 40

# Un píxel se considera parte de una "forma" (y no del fondo) cuando su
# distancia de color al fondo detectado supera este umbral.
_FOREGROUND_DISTANCE_THRESHOLD = 60.0


def build_glyph_templates(template_size: int = _TEMPLATE_SIZE) -> dict:
    """
    Renderiza todos los glifos de ALL_GLYPHS a `template_size` x
    `template_size` píxeles, para usarlos como plantillas de
    comparación ("template matching").
    """
    return {
        glyph: np.array(render_glyph(glyph[0], glyph[1], template_size), dtype=np.int32)
        for glyph in ALL_GLYPHS
    }


# Plantillas por defecto (tamaño estándar), calculadas una sola vez.
_DEFAULT_TEMPLATES = build_glyph_templates(_TEMPLATE_SIZE)


def _best_matching_glyph(module_gray: np.ndarray, templates: dict | None = None,
                          template_size: int = _TEMPLATE_SIZE) -> tuple[tuple[str, int], float]:
    """
    Devuelve (glifo_más_parecido, score_de_error). El score se usa,
    además de para elegir el glifo, para poder comparar la calidad
    global de dos decodificaciones candidatas (resolución de
    polaridad de colores en `_resolve_background_foreground`).
    """
    if templates is None:
        templates = _DEFAULT_TEMPLATES

    if module_gray.shape != (template_size, template_size):
        module_img = Image.fromarray(module_gray.astype(np.uint8))
        module_img = module_img.resize((template_size, template_size))
        module_gray = np.array(module_img)

    best_glyph = None
    best_score = None
    for glyph, template in templates.items():
        score = np.abs(module_gray.astype(np.int32) - template).sum()
        if best_score is None or score < best_score:
            best_score = score
            best_glyph = glyph

    return best_glyph, float(best_score)


def _decode_glyph_sequence(gray_array: np.ndarray, row_width: int, num_rows: int,
                            module_size: float, templates: dict | None = None,
                            template_size: int = _TEMPLATE_SIZE
                            ) -> tuple[list[tuple[str, int]], float]:
    """
    Recorre TODA la cuadrícula (incluidas las 3 celdas de esquina; el
    llamador es quien decide si las excluye por índice después) y
    devuelve (secuencia_de_glifos, error_total_acumulado).
    """
    glyph_sequence: list[tuple[str, int]] = []
    total_score = 0.0

    for row_index in range(num_rows):
        y0 = round((QUIET_ZONE + row_index) * module_size)
        y1 = round((QUIET_ZONE + row_index + 1) * module_size)
        for col_index in range(row_width):
            x0 = round((QUIET_ZONE + col_index) * module_size)
            x1 = round((QUIET_ZONE + col_index + 1) * module_size)
            module_gray = gray_array[y0:y1, x0:x1]
            if module_gray.size == 0:
                continue
            glyph, score = _best_matching_glyph(module_gray, templates, template_size)
            glyph_sequence.append(glyph)
            total_score += score

    return glyph_sequence, total_score


def _dominant_colors(rgb_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Detecta los DOS colores sólidos dominantes de la imagen (uno será
    el fondo, el otro el de las formas), a partir de qué colores
    exactos son más frecuentes en toda la imagen -- sin asumir nada
    sobre su posición.
    """
    flat = rgb_array.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(-counts)
    colors = colors[order].astype(np.float64)

    color_a = colors[0]
    color_b = None
    for candidate in colors[1:]:
        if np.linalg.norm(candidate - color_a) > _FOREGROUND_DISTANCE_THRESHOLD:
            color_b = candidate
            break

    if color_b is None:
        raise ValueError(
            "No se pudieron detectar dos colores distintos en la "
            "imagen: parece ser de un solo color. ¿Es realmente un "
            "Lancherix Visual Code?"
        )

    return color_a, color_b


def _to_relative_grayscale(rgb_array: np.ndarray, background_color: np.ndarray,
                            foreground_color: np.ndarray) -> np.ndarray:
    """
    Convierte la imagen RGB (con colores arbitrarios) a una escala de
    grises "relativa": 255 donde el píxel se parece al fondo, 0 donde
    se parece a las formas.
    """
    flat = rgb_array.reshape(-1, 3).astype(np.float64)
    distance_to_bg = np.linalg.norm(flat - background_color, axis=1)
    distance_to_fg = np.linalg.norm(flat - foreground_color, axis=1)
    denominator = distance_to_bg + distance_to_fg
    denominator[denominator == 0] = 1.0
    gray = 255.0 * (distance_to_fg / denominator)
    return gray.reshape(rgb_array.shape[:2])


def _resolve_background_foreground(
        rgb_array: np.ndarray, color_a: np.ndarray, color_b: np.ndarray,
        row_width: int, num_rows: int, module_size: float,
        templates: dict | None = None, template_size: int = _TEMPLATE_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int]]]:
    """
    Dados los dos colores sólidos dominantes, decide la polaridad
    correcta probando AMBAS asignaciones posibles, decodificando la
    cuadrícula completa con cada una, y quedándose con la que
    produzca el menor error total de coincidencia de plantillas.

    Returns: (background_color, foreground_color, gray_array,
              glyph_sequence) ya para la asignación ganadora.
    """
    best = None
    for background_color, foreground_color in ((color_a, color_b), (color_b, color_a)):
        gray_array = _to_relative_grayscale(rgb_array, background_color, foreground_color)
        glyph_sequence, total_score = _decode_glyph_sequence(
            gray_array, row_width, num_rows, module_size, templates, template_size)
        if best is None or total_score < best[0]:
            best = (total_score, background_color, foreground_color, gray_array, glyph_sequence)

    _, background_color, foreground_color, gray_array, glyph_sequence = best
    return background_color, foreground_color, gray_array, glyph_sequence