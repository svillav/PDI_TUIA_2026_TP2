"""
=====================================================================================
 Procesamiento de Imagenes I - TP N2 - Problema 1
 Deteccion y clasificacion de pastillas (pills.png)
-------------------------------------------------------------------------------------
 Pipeline (tecnicas simples vistas en clase, sin morfologia):
   A) Segmentar la cinta (ROI)   -> Umbralado (banda de gris automatica) + componente mayor
   B) Detectar cada pastilla     -> HSV (brillo|color) + componentes conectadas + filtro de area
   C) Clasificar (forma + color) -> Descriptores invariantes (U7) + Color HSV (U5)
   D) Informar por consola + imagen etiquetada
=====================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, ticks=False):
    if new_fig:
        plt.figure()
    plt.imshow(img) if color_img else plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if new_fig:
        plt.show(block=blocking)


# === CARGA DE IMAGEN ================================================================
imagen_bgr = cv2.imread('src/pills.png')
imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
tono, saturacion, valor = cv2.split(hsv)
ALTO, ANCHO = gris.shape
area_imagen = ALTO * ANCHO


# === A) SEGMENTAR LA CINTA (ROI) ====================================================
# Histograma + suavizado para detectar el pico (moda) de la cinta
# La cinta es la gran region de gris medio. Como ocupa casi toda la foto, es el pico
# (moda) del histograma: tomo la banda de grises alrededor de ese pico (mientras siga por
# encima del 15% de su altura). Asi el rango sale automatico, sin valores "a ojo".
histograma = cv2.calcHist([gris], [0], None, [256], [0, 256]).flatten()
histograma_suave = cv2.GaussianBlur(histograma, (1, 9), 0).flatten()
pico_gris = int(np.argmax(histograma_suave))

# Banda automática alrededor del pico
altura_corte = histograma_suave[pico_gris] * 0.15
gris_min = pico_gris
while gris_min > 0 and histograma_suave[gris_min] > altura_corte:
    gris_min -= 1
gris_max = pico_gris
while gris_max < 255 and histograma_suave[gris_max] > altura_corte:
    gris_max += 1

# Binarización por banda y selección de la componente más grande
mascara_gris = ((gris > gris_min) & (gris < gris_max)).astype(np.uint8) * 255
_, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(mascara_gris, connectivity=8)
idx_cinta = 1 + int(np.argmax(estadisticas[1:, cv2.CC_STAT_AREA]))
cinta = (etiquetas == idx_cinta).astype(np.uint8)

# Relleno de huecos con contornos
contornos, _ = cv2.findContours(cinta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
roi_cinta = np.zeros_like(cinta)
cv2.drawContours(roi_cinta, contornos, -1, 1, thickness=-1)

imshow(roi_cinta * 255, title="A) ROI de la cinta")


# === B) DETECTAR Y SEGMENTAR PASTILLAS ==============================================
# Máscara: pastillas son brillantes (blancas) o saturadas (coloreadas), dentro de la ROI
# El fondo es gris oscuro (valor~76, saturacion~4). La mitad azul de las capsulas es
# oscura -> la salva la saturacion. El umbral de brillo se deriva del fondo de la cinta.
umbral_brillo = int(valor[cinta == 1].mean() + 45)
mascara_pastillas = (
    ((valor > umbral_brillo) | (saturacion > 40)) & (roi_cinta == 1)
).astype(np.uint8) * 255

# Relleno de huecos internos
contornos, _ = cv2.findContours(mascara_pastillas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mascara_pastillas = np.zeros_like(mascara_pastillas)
cv2.drawContours(mascara_pastillas, contornos, -1, 255, thickness=-1)

# Componentes conectadas + filtrado por área proporcional
_, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(mascara_pastillas, connectivity=8)
area_min = 0.0003 * area_imagen
area_max = 0.05 * area_imagen
pastillas = [
    i for i in range(1, len(estadisticas))
    if area_min < estadisticas[i, cv2.CC_STAT_AREA] < area_max
]

imshow(mascara_pastillas, title=f"B) {len(pastillas)} pastillas segmentadas")


# === C) CLASIFICACIÓN POR FORMA Y COLOR =============================================
# FORMA (descriptores invariantes a escala/rotacion):
#   relacion_aspecto = lado mayor/menor del rect. rotado -> alargada (>1.75) = capsula
#   llenado_rect = area / area del rect. rotado -> cuadrada ~0.9 / redonda ~0.785
# COLOR (HSV dentro de la pastilla): saturacion baja=blanca; tono rojo/amarillo/azul=rosa/amarilla/azul
def clasificar_forma(id_pastilla):
    mascara = (etiquetas == id_pastilla).astype(np.uint8)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contorno = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(contorno)
    (_, _), (lado_a, lado_b), _ = cv2.minAreaRect(contorno)

    relacion_aspecto = max(lado_a, lado_b) / max(min(lado_a, lado_b), 1)
    llenado_rect = area / max(lado_a * lado_b, 1)

    if relacion_aspecto > 1.75:
        return 'Capsula'
    elif llenado_rect > 0.86:
        return 'Cuadrada'
    else:
        return 'Redonda'

def clasificar_color(id_pastilla):
    pixeles = (etiquetas == id_pastilla)
    tono_px = tono[pixeles]
    sat_px = saturacion[pixeles]

    if np.mean((tono_px >= 95) & (tono_px <= 135) & (sat_px >= 80)) > 0.10:
        return 'Azul'
    elif np.mean((tono_px >= 15) & (tono_px <= 40) & (sat_px >= 60)) > 0.30:
        return 'Amarilla'
    elif np.mean(((tono_px <= 10) | (tono_px >= 165)) & (sat_px >= 55)) > 0.30:
        return 'Rosa'
    else:
        return 'Blanca'

# Mapeo (forma, color) -> (código, color BGR para visualizar)
TIPOS = {
    ('Redonda', 'Blanca'): ('RB', (0, 180, 0)),
    ('Redonda', 'Rosa'): ('RR', (200, 0, 200)),
    ('Cuadrada', 'Blanca'): ('CB', (0, 140, 255)),
    ('Capsula', 'Amarilla'): ('CA', (0, 200, 200)),
    ('Capsula', 'Azul'): ('CZ', (255, 60, 0)),
}

DESCRIPCION = {
    'RB': 'Redonda Blanca',
    'RR': 'Redonda Rosa',
    'CB': 'Cuadrada Blanca',
    'CA': 'Capsula Amarilla',
    'CZ': 'Capsula Azul',
}

clasificacion = {}
conteo = Counter()
for id_pastilla in pastillas:
    forma = clasificar_forma(id_pastilla)
    color = clasificar_color(id_pastilla)
    codigo, _ = TIPOS.get((forma, color), (forma[0] + color[0], (0, 0, 255)))
    clasificacion[id_pastilla] = (forma, color, codigo)
    conteo[codigo] += 1


# === D) INFORME POR CONSOLA E IMAGEN ETIQUETADA =====================================
print("=" * 55)
print(" RESULTADOS - Detección y clasificación de pastillas")
print("=" * 55)
print(f" Total de pastillas detectadas: {len(pastillas)}")
print(f"\n Conteo por tipo:")
for codigo in sorted(conteo.keys()):
    descripcion = DESCRIPCION.get(codigo, codigo)
    print(f"   {codigo}  ({descripcion:18s}): {conteo[codigo]}")
print("=" * 55)

imagen_salida = imagen_bgr.copy()
contador_por_tipo = defaultdict(int)
escala_fuente = ANCHO / 1600.0
grosor_linea = max(1, int(2 * escala_fuente))

for id_pastilla in pastillas:
    forma, color, codigo = clasificacion[id_pastilla]
    _, color_bgr = TIPOS.get((forma, color), (codigo, (0, 0, 255)))
    contador_por_tipo[codigo] += 1
    etiqueta = f"{codigo}{contador_por_tipo[codigo]}"

    x = estadisticas[id_pastilla, cv2.CC_STAT_LEFT]
    y = estadisticas[id_pastilla, cv2.CC_STAT_TOP]
    w = estadisticas[id_pastilla, cv2.CC_STAT_WIDTH]
    h = estadisticas[id_pastilla, cv2.CC_STAT_HEIGHT]

    cv2.rectangle(imagen_salida, (x, y), (x + w, y + h), color_bgr, 2)
    cv2.putText(
        imagen_salida, etiqueta, (x, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX, escala_fuente, color_bgr, grosor_linea, cv2.LINE_AA
    )

imshow(
    cv2.cvtColor(imagen_salida, cv2.COLOR_BGR2RGB),
    color_img=True,
    title="D) Pastillas clasificadas"
)

cv2.imwrite('pills_resultado.png', imagen_salida)
plt.show(block=True)
