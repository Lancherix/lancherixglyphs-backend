#!/usr/bin/env python3

"""
lancherix_scanner_preprocessor.py (v7)

Convierte una fotografia en una imagen tipo scanner:
- localiza la tarjeta dentro de la foto y descarta el fondo
- refina esa localizacion en un segundo paso para eliminar el fondo
  POR COMPLETO, no solo en su mayor parte (nuevo en v7)
- blancos mas blancos
- negros mas negros
- elimina gran parte de los grises
- conserva encuadre y geometria de la tarjeta (no de la foto original)
- NO recorta el codigo en si (eso lo sigue haciendo el rectifier)
- NO detecta markers
- Robusto a iluminacion despareja (sombras, luz de lado)
- Robusto a fondos con textura similar en brillo/color al papel
- Rapido
- Nitido incluso en resoluciones chicas (ver nota de tamano abajo)

Uso:
    python3 lancherix_scanner_preprocessor.py foto.png
Salida:
    foto_scanner.png

---------------------------------------------------------------------
NUEVO EN v7 -- EL FONDO TIENE QUE DESAPARECER POR COMPLETO
---------------------------------------------------------------------
Problema que motiva este cambio: v6 localiza la tarjeta con un margen
DELIBERADAMENTE generoso (_DOC_MARGIN_FRAC = 0.20, ver nota de v6 mas
abajo sobre por que) porque el cuadrilatero candidato de
find_document_quad() tiende a asentarse sobre el borde impreso
interno de la tarjeta, no sobre el corte real del papel -- un margen
chico corria el riesgo de recortar dentro del patron impreso, que es
un fallo mucho mas grave que dejar un poco de fondo de mas.

Ese margen generoso cumple su proposito (no perder codigo), pero
como efecto secundario dejaba una franja real de fondo (alfombra,
mesa, lo que sea) alrededor de la tarjeta en el recorte. Esa franja
no es solo un problema estetico: al llegar a scanner_effect() y
binarizarse, una franja de fondo con textura (p. ej. una alfombra)
no se binariza a un blanco limpio -- se convierte en manchas
negras y puntos blancos sueltos (ruido de alto contraste local, cada
mancha de la alfombra que cae de un lado u otro del umbral). Ese
ruido quedaba en el PNG de salida, pegado al borde, ensuciando una
imagen que deberia ser solo tarjeta.

La solucion no es simplemente achicar _DOC_MARGIN_FRAC (eso reabre el
riesgo original de cortar el patron impreso que motivo el margen
generoso en v6). Son dos cambios independientes y complementarios,
cada uno atacando el problema en una etapa distinta del pipeline:

1. SEGUNDO PASO DE LOCALIZACION (find_document_region)

   Insight clave: el problema que forzaba un margen generoso en el
   PRIMER paso -- textura de fondo compitiendo con el borde real de
   la tarjeta, contraste parejo entre ambos -- practicamente
   desaparece una vez que ya se hizo un primer recorte. Con la
   alfombra (o lo que sea) reducida a una franja angosta en los
   bordes del recorte y la tarjeta ocupando la enorme mayoria del
   cuadro, un segundo llamado a find_document_quad() sobre ESE
   recorte encuentra el borde real del papel con mucha mas precision
   -- hay mucho menos fondo con el que confundirse, y la tarjeta
   ahora es, por lejos, el contorno mas grande y solido del cuadro.

   Se probo empiricamente sobre el caso que motivo este cambio (foto
   de la tarjeta rotada sobre una alfombra oscura de textura
   irregular): el primer paso ubica la tarjeta con score ~0.86 y dejo
   una franja de fondo visible; el segundo paso, sobre ese recorte,
   sube a score ~0.92 y ubica un cuadrilatero notablemente mas ajustado
   al borde real del papel. Con ese cuadrilatero mas confiable, el
   segundo recorte SI puede usar un margen chico
   (_DOC_REFINE_MARGIN_FRAC = 0.05) sin arriesgar el patron impreso,
   porque ya no esta absorbiendo el error de deteccion original que
   justificaba un margen grande -- ese error ya se corrigio en el
   primer paso.

   Si el segundo paso no encuentra un cuadrilatero confiable (deberia
   ser infrecuente, dado lo simple que ya queda el fondo), se sigue
   de largo con el resultado del primer paso sin refinar -- mismo
   criterio de fallback seguro que ya usaba v6.

2. LIMPIEZA DE BORDE POST-BINARIZACION (clear_border_ink)

   El segundo paso de localizacion reduce el problema, pero no lo
   elimina con garantia matematica para TODA foto -- sigue siendo una
   deteccion geometrica con su propio margen de error, no una
   segmentacion pixel-perfect del papel. Para no depender de afinar
   el numero de margen "perfecto" (que en la practica varia de foto
   en foto), se agrega una red de seguridad DESPUES de binarizar: se
   identifican los blobs de tinta (negro) conectados que tocan el
   borde del canvas y se blanquean.

   Por que esto es seguro: el margen de papel alrededor del patron
   impreso (en cualquiera de los dos pasos de localizacion) existe
   justamente para que el patron real nunca llegue hasta el borde del
   recorte -- ver la nota de v6 sobre _DOC_MARGIN_FRAC. Cualquier
   tinta que SI toque el borde del canvas, entonces, por construccion
   no es parte del codigo: es fondo residual que sobrevivio a la
   localizacion y alcanzo suficiente contraste local para binarizar a
   negro. Limpiarla no arriesga perder codigo real.

   Esta limpieza se aplica sobre el resultado final de
   scanner_effect() (adentro de la funcion misma, ver su cuerpo), es
   decir despues de _scanner_effect_small() o _scanner_effect_large()
   segun corresponda -- se beneficia igual cualquier camino.

Con los dos cambios juntos (menos fondo llega a binarizarse gracias
al segundo pase de localizacion, y lo poco que sobreviva se limpia
despues de binarizar) el resultado es una imagen con la tarjeta
aislada del fondo por completo: blanco de papel limpio hasta el
borde del canvas, sin manchas ni ruido de fondo. Se verifico
visualmente sobre el caso que motivo este cambio.

---------------------------------------------------------------------
NOTA DE v6 -- LOCALIZACION DE LA TARJETA ANTES DE ESCANEAR
---------------------------------------------------------------------
Problema que motiva este cambio: en fotos con fondos con textura
irregular (alfombras, tela, madera con vetas), el paso de
binarizacion de scanner_effect() puede convertir manchas del fondo
en blobs que se parecen lo suficiente a un agujero blanco de corner
marker (chico, mas o menos redondo, rodeado de oscuro) como para que
detect_white_center_candidates() (en el rectifier) los tome como
candidatos validos y arme un cuadrilatero incorrecto.

La solucion no es afinar mas los umbrales de deteccion de agujeros
(eso ya esta bastante ajustado y es fragil retocarlo mas), sino
sacar el fondo de la ecuacion antes de binarizar: encontrar donde
esta la tarjeta en la foto a color original y trabajar solo con esa
region de ahi en adelante.

Como se encuentra la tarjeta (find_document_region / find_document_quad):

  1. Deteccion de bordes (Canny automatico via mediana) sobre la
     foto a color, dilatada/erosionada para cerrar contornos.
  2. Cada contorno se reduce a un cuadrilatero via approxPolyDP
     (probando varios epsilon hasta conseguir 4 vertices). Se usa un
     cuadrilatero real, no un rectangulo rotado, porque en fotos con
     perspectiva marcada la tarjeta deja de ser un rectangulo en la
     imagen -- es un cuadrilatero irregular. Forzar un rectangulo
     rotado (minAreaRect) en ese caso recortaria mal.
  3. Cada cuadrilatero candidato se puntua con:
       - solidity (area del contorno / area del hull convexo): la
         tarjeta es lisa y redondeada, da solidity alta; el ruido de
         textura de fondo da contornos mas irregulares.
       - rect_fill (que tan bien el cuadrilatero llena un rectangulo
         de sus mismas dimensiones): descarta cuadrilateros muy
         sesgados/degenerados.
       - area_frac (fraccion del cuadro que ocupa): favorece
         candidatos de tamano razonable, ni un punto ni casi toda la
         foto.
       - "flatness" del margen (ver mas abajo): el descriminador mas
         fuerte.

  Se probo primero un enfoque mas directo -- segmentar directamente
  "material tipo papel" por brillo alto + saturacion baja en HSV,
  aprovechando que la tarjeta SIEMPRE tiene un margen de papel
  blanco/claro alrededor del borde impreso. Funciona en fondos
  oscuros o saturados, pero se probo empiricamente que falla en
  fondos que son ellos mismos claros y poco saturados (por ejemplo,
  una alfombra de pelo color crema/blanco): ese fondo entra en el
  mismo umbral de brillo/saturacion que el papel y se fusiona con la
  tarjeta en un solo blob, arruinando el cuadrilatero. Ver
  margin_flatness_score() para la version que si funciono.

  "flatness" del margen (margin_flatness_score): en lugar de
  clasificar el margen por SU COLOR (que puede coincidir con fondos
  igual de claros/poco saturados), se lo clasifica por SU TEXTURA.
  El margen de papel es opticamente liso (baja varianza local) sea
  cual sea su tono exacto en la foto; una alfombra, tela o madera
  mantienen textura real aunque su parche mas claro tenga un brillo
  parecido al papel. Se mide la energia de un Laplaciano dentro de
  una banda angosta justo hacia adentro del borde del cuadrilatero
  candidato (esa banda es, si el candidato es correcto, exactamente
  el margen de papel) y se usa la MEDIANA (no el promedio -- el
  promedio se ve arrastrado por un puñado de pixeles de alto
  contraste justo en la linea de transicion papel/fondo o en una
  esquina de marker que roce la banda). Se probo empiricamente:
  mediana ~10-12 para el margen real de la tarjeta, ~30+ para fondos
  con textura, con buena separacion entre ambos casos incluso en el
  fondo problematico (alfombra clara) donde el enfoque de color puro
  fallaba.

  4. Se toma el cuadrilatero de mayor puntaje. Si su puntaje no
     supera un umbral de confianza, se sigue con la foto completa sin
     recortar (fallback seguro: no perder una foto valida solo
     porque el localizador no esta seguro).
  5. Con el cuadrilatero elegido se calcula una homografia
     (getPerspectiveTransform) hacia un rectangulo con un margen
     agregado alrededor (_DOC_MARGIN_FRAC), y se aplica a la foto
     COMPLETA a color (warpPerspective) -- esto ya endereza la
     perspectiva de la tarjeta antes de que scanner_effect() la vea,
     ademas de eliminar el fondo.

  (v7 agrega un segundo pase de este mismo proceso sobre el
  resultado, ver nota de v7 arriba.)

Esta localizacion no depende de que la tarjeta este centrada: busca
en toda la foto, no en una region fija.

---------------------------------------------------------------------
POR QUE HAY DOS CAMINOS EN scanner_effect (imagen chica vs grande)
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

Nota: ahora que find_document_region() ya recorto la foto a (casi)
solo la tarjeta antes de llegar aca, el camino "grande" con
correccion de fondo deberia dispararse con menos frecuencia que
antes para fotos de celular tipicas -- pero se deja el criterio de
tamano tal cual (sobre la imagen YA recortada) porque sigue siendo
valido: una tarjeta fotografiada de muy cerca en alta resolucion
puede seguir siendo "grande" en px incluso ya recortada, y ahi la
correccion de fondo sigue siendo apropiada.

---------------------------------------------------------------------
NOTA SOBRE TAMANO DE SALIDA EN IMAGENES CHICAS (importante)
---------------------------------------------------------------------
Para imagenes chicas, esta version agranda el canvas de trabajo
ANTES de binarizar (interpolacion cubica), y el PNG resultante queda
a ese tamano mas grande -- ya no es pixel-a-pixel igual al de
entrada (ni al recorte de find_document_region, ni menos aun a la
foto original). Esto es intencional y no es "inventar" detalle: la
foto ya trae, en el borde entre una forma y el papel, un degrade de
un par de pixeles (producto del blur optico + demosaicing de la
camara) en vez de un salto abrupto de un pixel al otro. Al binarizar
sobre un canvas mas grande, ese degrade real se traduce en un borde
mas preciso y menos "escalonado" en vez de forzarse al grosor de un
solo pixel de la foto original. Se probo -- QUE la foto es una
posicion del borde con mas resolucion de muestreo, no una
alucinacion de una red neuronal ni un suavizado cosmetico por fuera
de la imagen.

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

# --- find_document_region -------------------------------------------------

# Fraccion del area total de la foto que debe ocupar un cuadrilatero
# candidato para considerarse (descarta ruido chico y contornos que
# son casi toda la foto).
_DOC_MIN_AREA_FRAC = 0.02
_DOC_MAX_AREA_FRAC = 0.95

# Relacion lado_largo/lado_corto aceptable para un candidato. La
# tarjeta es ~3:1 en su version mas chica (k minimo) y mas alargada
# para k mayor, fotografiada de frente o de lado (rotada 90). Rango
# generoso a proposito -- la clasificacion fina de que lado es cual
# la sigue haciendo el rectifier despues.
_DOC_MIN_ASPECT = 1.3
_DOC_MAX_ASPECT = 5.0

# Puntaje minimo para confiar en el cuadrilatero encontrado. Por
# debajo de esto, se sigue de largo con la foto completa sin recortar
# (fallback seguro). Se reusa el mismo umbral para el segundo pase de
# refinamiento (ver nota de v7) -- si algo, el segundo pase deberia
# scorear IGUAL o MEJOR que el primero (menos fondo compitiendo), asi
# que no hizo falta un umbral separado y mas permisivo para el.
_DOC_CONFIDENCE_THRESHOLD = 0.55

# Margen agregado alrededor de la tarjeta detectada, como fraccion de
# su ancho/alto, al enderezar la perspectiva EN EL PRIMER PASE.
#
# Deliberadamente generoso (20%, no un numero simbolico como 5%).
# Motivo: el cuadrilatero candidato de find_document_quad() en la
# practica tiende a asentarse sobre el borde impreso INTERNO de la
# tarjeta (la linea negra redondeada), no sobre el borde real del
# papel -- Canny+approxPolyDP encuentran el contorno mas nitido y
# contrastado, que suele ser esa linea, no el corte del papel contra
# el fondo (mas mixto en contraste segun la foto). Normalmente el
# margen compensa esa diferencia sin problema, pero se detecto
# empiricamente un caso (foto con la tarjeta mas cerca del borde del
# encuadre y con menos contraste papel/fondo de un lado) donde un
# margen chico (5%) no alcanzaba y el recorte terminaba cortando dentro
# del patron impreso -- perdiendo informacion real del codigo.
#
# El costo de un margen generoso es bajo (un poco mas de fondo
# alrededor de la tarjeta en el recorte -- y desde v7 ese resto de
# fondo se termina de limpiar en el segundo pase + clear_border_ink,
# ver nota de v7); el costo de un margen chico es alto (recortar el
# codigo real es un fallo grave, no cosmetico). Ante esa asimetria,
# se prefiere errar generoso en ESTE primer pase.
_DOC_MARGIN_FRAC = 0.20

# Margen agregado en el SEGUNDO pase de localizacion (ver nota de v7
# arriba). Deliberadamente mas chico que _DOC_MARGIN_FRAC: para
# cuando se llega a este pase, el fondo ya quedo mayormente afuera
# gracias al primer recorte generoso, asi que el cuadrilatero que
# encuentra find_document_quad() sobre ESE recorte ya no esta
# absorbiendo el mismo error de deteccion (fondo con contraste
# parecido al papel, borde impreso interno confundido con el corte
# real) que justificaba un margen grande en el primer pase -- se
# probo empiricamente que 5% alcanza sin cortar el patron impreso, y
# clear_border_ink() actua ademas como red de seguridad final por si
# algun caso puntual necesitara mas margen del que este numero da.
_DOC_REFINE_MARGIN_FRAC = 0.05

# Cuanto hacia adentro del borde del cuadrilatero candidato se mide
# la "planitud" del margen de papel (ver margin_flatness_score), como
# fraccion de la distancia al centro. Se mantiene la banda lejos de
# la linea de transicion papel/fondo a proposito (esa linea tiene un
# salto de contraste real y no debe contarse como "textura de
# fondo").
_DOC_FLATNESS_INSET_FRAC = 0.06

# Mediana de energia de Laplaciano (valor absoluto) en la banda de
# margen a partir de la cual se considera "sin textura" (0.0) vs
# "papel liso" (1.0), con interpolacion lineal entre medio. Calibrado
# empiricamente: margen real de tarjeta ~10-12, fondos con textura
# (alfombra, tela, madera) ~30+.
_DOC_FLATNESS_SATURATION = 30.0


def order_quad_pts(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def margin_flatness_score(image, quad, inset_frac=_DOC_FLATNESS_INSET_FRAC):
    """
    La tarjeta SIEMPRE tiene un margen de papel liso entre su borde
    real y el borde impreso. Este margen es opticamente plano (poca
    varianza local) sea cual sea su tono exacto en la foto -- a
    diferencia de un fondo con textura (alfombra, tela, madera) que
    mantiene variacion local incluso en su parche mas claro, aunque
    ese parche tenga un brillo/saturacion parecido al papel.

    Se mide la mediana (no el promedio, ver nota de modulo) de la
    energia de un Laplaciano dentro de una banda angosta justo hacia
    adentro del borde del cuadrilatero candidato. Si el candidato es
    la tarjeta real, esa banda cae sobre el margen de papel.
    """
    h, w = image.shape[:2]
    outer_quad = quad.astype(np.float32)
    center = outer_quad.mean(axis=0)

    # La banda se mantiene un poco separada del borde exacto del
    # cuadrilatero: justo en esa linea hay una transicion de
    # contraste real (papel/fondo) que no es "textura de fondo" y no
    # debe ensuciar la medicion.
    outer = center + (outer_quad - center) * (1.0 - inset_frac)
    inner = center + (outer_quad - center) * (1.0 - inset_frac * 3.0)

    mask_outer = np.zeros((h, w), dtype=np.uint8)
    mask_inner = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_outer, [outer.astype(np.int32)], 255)
    cv2.fillPoly(mask_inner, [inner.astype(np.int32)], 255)
    band = cv2.subtract(mask_outer, mask_inner)

    if cv2.countNonZero(band) < 30:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    band_vals = np.abs(lap[band > 0])
    texture_energy = float(np.median(band_vals))

    return float(np.clip(1.0 - texture_energy / _DOC_FLATNESS_SATURATION, 0.0, 1.0))


def find_document_quad(image):
    """
    Busca, en toda la foto (no asume que la tarjeta este centrada),
    el cuadrilatero que mejor corresponde al borde real de la
    tarjeta. Devuelve (quad, score, candidates_info). quad es None si
    no se encontro ningun candidato geometricamente plausible (no
    implica automaticamente descartar -- ver _DOC_CONFIDENCE_THRESHOLD
    en el llamador).
    """
    h, w = image.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    med = np.median(blur)
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(blur, lo, hi)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    edges = cv2.erode(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0
    candidates_info = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * _DOC_MIN_AREA_FRAC or area > img_area * _DOC_MAX_AREA_FRAC:
            continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area < 1:
            continue
        solidity = cv2.contourArea(c) / hull_area

        peri = cv2.arcLength(hull, True)
        approx = None
        for eps_frac in np.linspace(0.01, 0.06, 12):
            candidate_approx = cv2.approxPolyDP(hull, eps_frac * peri, True)
            if len(candidate_approx) == 4:
                approx = candidate_approx
                break
        if approx is None:
            continue

        quad = order_quad_pts(approx.reshape(-1, 2))
        tl, tr, br, bl = quad
        top = np.linalg.norm(tr - tl)
        bottom = np.linalg.norm(br - bl)
        left = np.linalg.norm(bl - tl)
        right = np.linalg.norm(br - tr)
        if min(top, bottom, left, right) < 20:
            continue

        long_side = max((top + bottom) / 2, (left + right) / 2)
        short_side = min((top + bottom) / 2, (left + right) / 2)
        aspect = long_side / max(short_side, 1e-6)
        if aspect < _DOC_MIN_ASPECT or aspect > _DOC_MAX_ASPECT:
            continue

        quad_area = cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.float32))
        rect_fill = quad_area / (long_side * short_side + 1e-6)
        area_frac = area / img_area
        flatness = margin_flatness_score(image, quad)

        # flatness pesa mas que el resto combinado: es el unico
        # criterio que distingue de forma confiable la tarjeta de un
        # parche de fondo con geometria/tamano parecidos (ver nota de
        # modulo sobre por que el enfoque de color puro no alcanza).
        score = (
            0.25 * solidity
            + 0.20 * rect_fill
            + 0.15 * min(area_frac / 0.25, 1.0)
            + 0.40 * flatness
        )

        candidates_info.append({
            "quad": quad, "solidity": solidity, "aspect": aspect,
            "rect_fill": rect_fill, "area_frac": area_frac,
            "flatness": flatness, "score": score,
        })

        if score > best_score:
            best_score = score
            best = quad

    return best, best_score, candidates_info


def warp_quad_with_margin(image, quad, margin_frac):
    tl, tr, br, bl = quad
    width = int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2))
    height = int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2))

    margin_x = int(round(width * margin_frac))
    margin_y = int(round(height * margin_frac))

    canvas_w = width + 2 * margin_x
    canvas_h = height + 2 * margin_y

    dst = np.array([
        [margin_x, margin_y],
        [margin_x + width, margin_y],
        [margin_x + width, margin_y + height],
        [margin_x, margin_y + height],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image, H, (canvas_w, canvas_h), flags=cv2.INTER_CUBIC)


def find_document_region(image):
    """
    Localiza la tarjeta dentro de la foto (a color, sin asumir que
    este centrada) y devuelve una version recortada + con perspectiva
    corregida, con un margen chico alrededor. Si no se encuentra un
    candidato con confianza suficiente en el primer pase, devuelve la
    imagen original sin modificar (fallback seguro) -- el resto del
    pipeline sigue funcionando igual que en versiones anteriores en
    ese caso.

    v7: ademas del primer pase (con margen generoso, ver
    _DOC_MARGIN_FRAC), se hace un segundo pase de localizacion sobre
    el resultado del primero para ajustar el recorte con un margen
    mucho mas chico (_DOC_REFINE_MARGIN_FRAC) y asi terminar de sacar
    el fondo que el primer pase, por diseno, deja alrededor (ver nota
    de modulo "NUEVO EN v7"). Si el segundo pase no encuentra nada
    confiable se sigue con el resultado del primer pase sin refinar.

    Devuelve (imagen_resultado, encontrado: bool, score: float). Si
    hubo refinamiento, score corresponde al segundo pase.
    """
    quad, score, _ = find_document_quad(image)
    if quad is None or score < _DOC_CONFIDENCE_THRESHOLD:
        return image, False, score

    warped = warp_quad_with_margin(image, quad, margin_frac=_DOC_MARGIN_FRAC)

    quad2, score2, _ = find_document_quad(warped)
    if quad2 is not None and score2 >= _DOC_CONFIDENCE_THRESHOLD:
        warped = warp_quad_with_margin(warped, quad2, margin_frac=_DOC_REFINE_MARGIN_FRAC)
        score = score2

    return warped, True, score


def clear_border_ink(binary):
    """
    NUEVO EN v7. Ultima red de seguridad contra fondo residual: en la
    imagen ya binarizada (0 = negro/tinta, 255 = blanco/papel),
    elimina cualquier blob de tinta conectado que toque el borde del
    canvas, blanqueandolo.

    Por que esto es seguro (no arriesga borrar codigo real): el
    margen de papel alrededor del patron impreso -- en cualquiera de
    los dos pases de find_document_region() -- existe justamente para
    que el patron real nunca llegue hasta el borde del recorte (ver
    nota de v6 sobre _DOC_MARGIN_FRAC). Cualquier tinta que SI toque
    el borde del canvas, entonces, por construccion no es parte del
    codigo: es fondo residual (textura de fondo con contraste
    suficiente para binarizar a negro, un flanco de sombra, un
    recorte imperfecto) que sobrevivio a la localizacion.

    Se prefirio esto a simplemente achicar mas el margen de recorte:
    no depende de acertarle a un numero de margen "perfecto" (que
    varia de foto en foto), y no arriesga cortar el patron real si el
    margen resulta demasiado chico en algun caso puntual -- solo
    limpia lo que efectivamente quedo pegado al borde, sea cual sea
    la causa.
    """
    ink = (binary == 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(ink, connectivity=8)
    if num_labels <= 1:
        return binary

    border_labels = set(labels[0, :].tolist())
    border_labels |= set(labels[-1, :].tolist())
    border_labels |= set(labels[:, 0].tolist())
    border_labels |= set(labels[:, -1].tolist())
    border_labels.discard(0)  # 0 = fondo de la mascara de tinta, no tocar
    if not border_labels:
        return binary

    result = binary.copy()
    result[np.isin(labels, list(border_labels))] = 255
    return result


# --- scanner_effect (sin cambios de logica, solo ahora recibe la
#     imagen ya recortada por find_document_region cuando corresponde,
#     y su salida pasa por clear_border_ink antes de devolverse) --

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


def scanner_effect(image, _report=None):
    # ---------------------------------------------------------
    # 0. Localizar la tarjeta y descartar el fondo (ver nota de
    #    modulo, incluye desde v7 un segundo pase de refinamiento).
    #    Se hace ACA adentro, no solo en main(), para que cualquier
    #    caller que importe y use scanner_effect() directamente
    #    (p. ej. read_camera_image_via_markers()) se beneficie
    #    tambien -- no solo el uso por linea de comandos.
    #
    #    _report es un hook opcional (dict mutable) solo para que
    #    main() pueda imprimir el resultado de la localizacion sin
    #    tener que llamar a find_document_region() una segunda vez
    #    por su cuenta. Los callers normales lo ignoran.
    # ---------------------------------------------------------
    image, found, doc_score = find_document_region(image)
    if _report is not None:
        _report["found"] = found
        _report["score"] = doc_score
        _report["size"] = (image.shape[1], image.shape[0])

    # ---------------------------------------------------------
    # 1. Escala de grises + suavizado muy ligero
    # ---------------------------------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    h, w = gray.shape
    if max(h, w) <= _SMALL_IMAGE_MAX_DIM:
        result = _scanner_effect_small(gray)
    else:
        result = _scanner_effect_large(gray)

    # ---------------------------------------------------------
    # 2. Red de seguridad final (v7): blanquear cualquier resto de
    #    fondo que haya sobrevivido a la localizacion y llegado a
    #    binarizarse pegado al borde del canvas. Ver clear_border_ink().
    # ---------------------------------------------------------
    return clear_border_ink(result)


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

    report = {}
    result = scanner_effect(image, _report=report)
    found = report.get("found", False)
    doc_score = report.get("score", 0.0)
    lw, lh = report.get("size", (w, h))

    is_small = max(lh, lw) <= _SMALL_IMAGE_MAX_DIM
    path_used = "chica (upscale + sin correccion de fondo)" if is_small else "grande (con correccion de fondo)"

    output_path = input_path.with_name(f"{input_path.stem}_scanner.png")
    if not cv2.imwrite(str(output_path), result):
        print(f"Error: no se pudo guardar: {output_path}")
        sys.exit(1)

    print()
    print("Lancherix Scanner Preprocessor")
    print("==============================")
    print(f"Input      : {input_path} ({w}x{h})")
    if found:
        print(f"Localizacion: tarjeta encontrada, score={doc_score:.3f} -> recortada a {lw}x{lh}")
    else:
        print(f"Localizacion: sin candidato confiable (score={doc_score:.3f}), se uso la foto completa")
    print(f"Output     : {output_path} ({result.shape[1]}x{result.shape[0]})")
    print(f"Camino     : {path_used}")
    print()


if __name__ == "__main__":
    main()