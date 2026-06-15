"""
=====================================================================================
 Procesamiento de Imagenes I - TP N2 - Problema 2
 Deteccion de placas patente y segmentacion de caracteres (img_1.jpg ... img_12.jpg)
-------------------------------------------------------------------------------------
 Estrategia:
   A) Detectar/segmentar la patente:
        - Umbral adaptativo (Ayuda #2) -> robusto a iluminacion variable
        - Candidatos con forma de caracter por proporciones (Ayuda #1: mas altos que anchos)
        - Agrupar los caracteres alineados y contiguos -> la fila de la patente
   B) Segmentar cada caracter dentro de la patente recortada.
 Todos los umbrales geometricos son proporciones (relativas al tamaño de la imagen o
 del caracter), no pixeles fijos -> robusto a la distancia/escala de cada foto.
=====================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from collections import defaultdict

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, ticks=False):
    if new_fig:
        plt.figure()
    plt.imshow(img) if color_img else plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if new_fig:
        plt.show(block=blocking)

# === FUNCIONES AUXILIARES ===========================================================
def detectar_candidatos(gris):
    """Detecta blobs con forma de carácter en una imagen en grises (Ayuda #1 + #2)."""
    alto, ancho = gris.shape
    tam_bloque = max(11, (min(alto, ancho) // 20) | 1)
    
    binaria = cv2.adaptiveThreshold(
        gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, tam_bloque, 10
    )
    
    n, etiquetas, datos, centroides = cv2.connectedComponentsWithStats(binaria, 8)
    
    candidatos = []
    for i in range(1, n):
        x, y, w, h, area = datos[i]
        if w == 0 or h == 0:
            continue
        if 1.1 <= h/w <= 3.0 and 0.012 <= h/alto <= 0.18 and area >= 0.3 * w * h:
            candidatos.append((x, y, w, h, centroides[i][0], centroides[i][1]))

    return binaria, candidatos

def agrupar(candidatos):
    """Agrupa caracteres alineados, de altura similar y contiguos (Union-Find)."""
    n = len(candidatos)
    padre = list(range(n))
    
    def raiz(a):
        while padre[a] != a:
            padre[a] = padre[padre[a]]
            a = padre[a]
        return a
    
    for i in range(n):
        _, _, _, hi, cxi, cyi = candidatos[i]
        for j in range(i + 1, n):
            _, _, _, hj, cxj, cyj = candidatos[j]
            alto_ref = max(hi, hj)
            if (min(hi, hj) / alto_ref > 0.6 and
                abs(cyi - cyj) < 0.5 * alto_ref and
                abs(cxi - cxj) < 2.0 * alto_ref):
                padre[raiz(i)] = raiz(j)
    
    grupos = defaultdict(list)
    for i in range(n):
        grupos[raiz(i)].append(i)
    
    return [g for g in grupos.values() if len(g) >= 4]

def caja_grupo(candidatos, grupo):
    """Calcula la caja envolvente de un grupo."""
    xs = [candidatos[k][0] for k in grupo]
    ys = [candidatos[k][1] for k in grupo]
    xe = [candidatos[k][0] + candidatos[k][2] for k in grupo]
    ye = [candidatos[k][1] + candidatos[k][3] for k in grupo]
    return min(xs), min(ys), max(xe), max(ye)

def puntaje_patente(candidatos, grupo):
    """Asigna un puntaje al grupo según qué tan parecido es a una patente."""
    x0, y0, x1, y1 = caja_grupo(candidatos, grupo)
    aspecto = (x1 - x0) / max(y1 - y0, 1)
    n = len(grupo)
    
    if not (5 <= n <= 9 and 2.5 <= aspecto <= 7):
        return -1    
    return 100 - abs(n - 7) * 5 - abs(aspecto - 4) * 3

def segmentar_caracteres(recorte_gris, alto_char):
    """Segmenta caracteres individuales dentro del recorte de patente."""
    # Re-binarización local
    patente_bin = cv2.adaptiveThreshold(
        recorte_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, max(11, alto_char | 1), 10
    )    
    # Detección de blobs y filtros geométricos
    n, etq, datos_c, _ = cv2.connectedComponentsWithStats(patente_bin, 8)
    
    caracteres = []
    for i in range(1, n):
        x, y, w, h, area = datos_c[i]
        if w == 0:
            continue
        if (0.55 * alto_char <= h <= 1.5 * alto_char and
            h >= 0.9 * w and
            area >= 0.2 * w * h):
            caracteres.append((x, y, w, h))
    
    # Filtro por alineación al renglón dominante (mediana)
    if caracteres:
        centros_y = [y + h/2 for (x, y, w, h) in caracteres]
        alturas = [h for (x, y, w, h) in caracteres]
        cy_med = np.median(centros_y)
        h_med = np.median(alturas)

        caracteres = [
            (x, y, w, h) for (x, y, w, h) in caracteres
            if abs((y + h/2) - cy_med) < 0.5 * h_med
            and 0.7 * h_med <= h <= 1.4 * h_med
        ]
    
    # Ordenar de izquierda a derecha (por coordenada X)
    caracteres.sort(key=lambda c: c[0])
    
    # Fusionar cajas muy solapadas en X (caracteres partidos por la binarización)
    fusionados = []
    for c in caracteres:
        if fusionados:
            px, py, pw, ph = fusionados[-1]
            solape_x = min(px + pw, c[0] + c[2]) - max(px, c[0])
            if solape_x > 0.4 * min(pw, c[2]):
                # Solapan -> fusionamos en una sola caja envolvente
                nx0, ny0 = min(px, c[0]), min(py, c[1])
                nx1, ny1 = max(px + pw, c[0] + c[2]), max(py + ph, c[1] + c[3])
                fusionados[-1] = (nx0, ny0, nx1 - nx0, ny1 - ny0)
                continue
        fusionados.append(tuple(int(v) for v in c))    
    
    caracteres = fusionados
    
    # Descartar slivers (anchos muy chicos respecto a la mediana)
    if caracteres:
        ancho_med = np.median([w for (_, _, w, _) in caracteres])
        caracteres = [c for c in caracteres if c[2] >= 0.3 * ancho_med]

    return patente_bin, caracteres


# === BUCLE PRINCIPAL ======================================

# Buscar todas las imágenes en la carpeta patentes/
archivos = sorted(
    glob.glob('src/patentes/img_*.jpg'),
    key=lambda p: int(p.split('_')[1].split('.')[0])
)

print(f"\n{'='*60}")
print(f" Procesando {len(archivos)} imágenes de patentes")
print(f"{'='*60}\n")

for ruta in archivos:
    nro = int(ruta.split('_')[1].split('.')[0])
    
    # Cargar y convertir a grises
    img = cv2.imread(ruta)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ALTO, ANCHO = gris.shape
    
    # === A) DETECCIÓN DE PATENTE ====================================================
    _, candidatos = detectar_candidatos(gris)
    grupos = agrupar(candidatos)
    
    validos = [(puntaje_patente(candidatos, g), g) for g in grupos]
    validos = [(p, g) for p, g in validos if p > 0]
    
    if not validos:
        print(f"img_{nro:2d}:  No se detectó patente")
        continue
    
    _, mejor = max(validos, key=lambda t: t[0])
    x0, y0, x1, y1 = caja_grupo(candidatos, mejor)
    
    # Recortar con margen proporcional
    alto_char = int(np.median([candidatos[k][3] for k in mejor]))
    mx = int(0.6 * alto_char)
    rx0, ry0 = max(0, x0 - mx), max(0, y0 - mx)
    rx1, ry1 = min(ANCHO, x1 + mx), min(ALTO, y1 + mx)
    recorte_color = img[ry0:ry1, rx0:rx1]
    recorte_gris = gris[ry0:ry1, rx0:rx1]
    
    # === B) SEGMENTACIÓN DE CARACTERES ==============================================
    patente_bin, caracteres = segmentar_caracteres(recorte_gris, alto_char)
    
    # Dibujar caracteres detectados sobre el recorte
    recorte_viz = recorte_color.copy()
    for (x, y, w, h) in caracteres:
        cv2.rectangle(recorte_viz, (x, y), (x + w, y + h),
                      (0, 255, 255), max(1, alto_char // 20))
    
    print(f"img_{nro:2d}: patente en [{x0},{y0},{x1},{y1}] | {len(caracteres)} caracteres segmentados")
    
    # === VISUALIZACIÓN: 3 etapas por imagen =========================================
    img_det = img.copy()
    cv2.rectangle(img_det, (x0, y0), (x1, y1), (0, 255, 0), max(2, ANCHO // 400))
    
    plt.figure(figsize=(13, 5))
    plt.suptitle(f"Patente - img_{nro}")
    plt.subplot(1, 3, 1)
    imshow(cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB),
           new_fig=False, color_img=True, title="1) Patente detectada")
    plt.subplot(1, 3, 2)
    imshow(patente_bin, new_fig=False, title="2) Patente binarizada")
    plt.subplot(1, 3, 3)
    imshow(cv2.cvtColor(recorte_viz, cv2.COLOR_BGR2RGB),
           new_fig=False, color_img=True,
           title=f"3) {len(caracteres)} caracteres")
    plt.tight_layout()
    plt.show(block=False)

plt.show(block=True)