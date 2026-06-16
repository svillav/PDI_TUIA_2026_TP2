"""
=====================================================================================
 Procesamiento de Imagenes I - TP N2 - Problema 2
 Deteccion de placas patente y segmentacion de caracteres
-------------------------------------------------------------------------------------
 Estrategia:
   A) Detectar/segmentar la patente:
        - Umbral adaptativo -> robusto a iluminacion variable
        - Candidatos con forma de caracter por proporciones (Ayuda #1)
        - Dilatacion horizontal para unir los caracteres alineados
        - Componentes conectadas sobre la imagen dilatada
        - Validacion: aspecto, cantidad de candidatos y cobertura horizontal
   B) Segmentar cada caracter dentro de la patente recortada.
=====================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


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
    """
    Detecta blobs con forma de carácter.
    Umbral adaptativo gaussiano + componentes conectadas + filtros.
    """
    alto, ancho = gris.shape
    tam_bloque = max(11, (min(alto, ancho) // 20) | 1)
    
    binaria = cv2.adaptiveThreshold(
        gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, tam_bloque, 10
    )
    
    n, _, datos, centroides = cv2.connectedComponentsWithStats(binaria, 8)
    
    candidatos = []
    for i in range(1, n):
        x, y, w, h, area = datos[i]
        if w == 0 or h == 0:
            continue
        if 1.1 <= h/w <= 3.0 and 0.012 <= h/alto <= 0.18 and area >= 0.3 * w * h:
            candidatos.append((x, y, w, h, centroides[i][0], centroides[i][1]))

    return binaria, candidatos

def estimar_alto_caracter(candidatos):
    """
    Estima la altura tipica del caracter usando la mediana de las alturas.
    """
    alturas = np.array([h for (_, _, _, h, _, _) in candidatos])
    if len(alturas) == 0:
        return 20
    return int(np.median(alturas))

def detectar_patente(binaria, candidatos, alto_img):
    """
    Detecta la patente uniendo los caracteres alineados mediante dilatacion
    horizontal y componentes conectadas sobre la imagen dilatada.

    Validaciones:
      - Aspecto post-dilatacion 2.5-9 (la dilatacion infla el ancho real)
      - Cantidad de candidatos 5-9 -> patente tiene 7
      - Cobertura horizontal >= 40% -> caracteres distribuidos uniformemente
      - Altura razonable respecto a la imagen
    """
    if not candidatos:
        return None, None, None

    # Altura tipica usando percentil 75 (robusta a ruido de menor tamaño)
    alto_char_est = estimar_alto_caracter(candidatos)

    # Mascara solo con los candidatos validos
    mascara_cand = np.zeros_like(binaria)
    for (x, y, w, h, _, _) in candidatos:
        mascara_cand[y:y+h, x:x+w] = 255

    # === Dilatacion horizontal ===
    # SE rectangular de ancho ~1.8x altura del caracter, alto = 1.
    # Une caracteres de la misma fila sin invadir filas distintas.
    ancho_se = max(15, int(alto_char_est * 1.8))
    se_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (ancho_se, 1))
    dilatada = cv2.dilate(mascara_cand, se_horizontal)

    # === Componentes conectadas sobre la dilatada ===
    n, etq_grupos, datos_grupos, _ = cv2.connectedComponentsWithStats(dilatada, 8)

    # Pre-calculamos a que blob pertenece cada candidato
    candidatos_por_grupo = [[] for _ in range(n)]
    for idx, (_, _, _, _, cx, cy) in enumerate(candidatos):
        id_grupo = etq_grupos[int(cy), int(cx)]
        if id_grupo > 0:
            candidatos_por_grupo[id_grupo].append(idx)

    # === Validacion y seleccion del mejor blob ===
    mejor = None
    mejor_puntaje = -1
    mejor_grupo_idxs = None

    for i in range(1, n):
        x, y, w, h, _ = datos_grupos[i]
        if h == 0:
            continue
        aspecto = w / h
        n_cands = len(candidatos_por_grupo[i])

        # Filtros estructurales.
        # Nota: el aspecto post-dilatacion es mayor que el real (la dilatacion
        # horizontal infla el ancho), por eso el techo es 9 en lugar de 7.
        if not (2.5 <= aspecto <= 9.0):
            continue
        if not (5 <= n_cands <= 9):
            continue
        if not (0.02 <= h / alto_img <= 0.25):
            continue

        # Cobertura horizontal: los candidatos deben distribuirse uniformemente
        ancho_cubierto = sum(candidatos[k][2] for k in candidatos_por_grupo[i])
        cobertura = ancho_cubierto / w
        if cobertura < 0.40:
            continue

        # Puntaje: cuanto mas cerca de 7 candidatos y aspecto 4, mejor
        puntaje = 100 - abs(n_cands - 7) * 5 - abs(aspecto - 4) * 3
        if puntaje > mejor_puntaje:
            mejor_puntaje = puntaje
            mejor = (x, y, w, h)
            mejor_grupo_idxs = candidatos_por_grupo[i]

    if mejor is None:
        return None, None, None

    # Recalcular alto_char usando solo los caracteres del grupo ganador
    alto_char_grupo = int(np.median([candidatos[k][3] for k in mejor_grupo_idxs]))

    return mejor, alto_char_grupo, mejor_grupo_idxs


def segmentar_caracteres(recorte_gris, alto_char):
    """
    Segmenta caracteres individuales dentro del recorte de la patente.
    Re-binariza localmente y filtra blobs por forma y alineacion.
    """
    patente_bin = cv2.adaptiveThreshold(
        recorte_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, max(11, alto_char | 1), 10
    )

    n, _, datos_c, _ = cv2.connectedComponentsWithStats(patente_bin, 8)

    caracteres = []
    for i in range(1, n):
        x, y, w, h, area = datos_c[i]
        if w == 0:
            continue
        if (0.55 * alto_char <= h <= 1.5 * alto_char
                and h >= 0.9 * w
                and area >= 0.2 * w * h):
            caracteres.append((x, y, w, h))

    # Filtro por fila dominante usando mediana
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

    caracteres.sort(key=lambda c: c[0])

    # Fusionar cajas solapadas en X (caracter partido por la binarizacion)
    fusionados = []
    for c in caracteres:
        if fusionados:
            px, py, pw, ph = fusionados[-1]
            solape_x = min(px + pw, c[0] + c[2]) - max(px, c[0])
            if solape_x > 0.4 * min(pw, c[2]):
                nx0, ny0 = min(px, c[0]), min(py, c[1])
                nx1, ny1 = max(px + pw, c[0] + c[2]), max(py + ph, c[1] + c[3])
                fusionados[-1] = (nx0, ny0, nx1 - nx0, ny1 - ny0)
                continue
        fusionados.append(tuple(int(v) for v in c))
    caracteres = fusionados

    # Descartar slivers: fragmentos muy angostos respecto a la mediana
    if caracteres:
        ancho_med = np.median([w for (_, _, w, _) in caracteres])
        caracteres = [c for c in caracteres if c[2] >= 0.3 * ancho_med]

    return patente_bin, caracteres


# === BUCLE PRINCIPAL ================================================================

archivos = sorted(
    glob.glob('src/patentes/img_*.jpg'),
    key=lambda p: int(p.split('_')[1].split('.')[0])
)

print(f"\n{'='*60}")
print(f" Procesando {len(archivos)} imágenes de patentes")
print(f"{'='*60}\n")

for ruta in archivos:
    nro = int(ruta.split('_')[1].split('.')[0])
    img = cv2.imread(ruta)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ALTO, ANCHO = gris.shape

    binaria, candidatos = detectar_candidatos(gris)
    bbox_patente, alto_char, _ = detectar_patente(binaria, candidatos, ALTO)

    if bbox_patente is None:
        print(f"img_{nro:2d}: No se detectó patente")
        continue

    x0, y0, w0, h0 = bbox_patente
    x1, y1 = x0 + w0, y0 + h0

    mx = int(0.6 * alto_char)
    rx0, ry0 = max(0, x0 - mx), max(0, y0 - mx)
    rx1, ry1 = min(ANCHO, x1 + mx), min(ALTO, y1 + mx)
    recorte_color = img[ry0:ry1, rx0:rx1]
    recorte_gris = gris[ry0:ry1, rx0:rx1]

    patente_bin, caracteres = segmentar_caracteres(recorte_gris, alto_char)

    recorte_viz = recorte_color.copy()
    for (x, y, w, h) in caracteres:
        cv2.rectangle(recorte_viz, (x, y), (x + w, y + h),
                      (0, 255, 255), max(1, alto_char // 20))

    print(f"img_{nro:2d}: patente en [{x0},{y0},{x1},{y1}] "
          f"| {len(caracteres)} caracteres segmentados")

    img_det = img.copy()
    cv2.rectangle(img_det, (x0, y0), (x1, y1), (0, 255, 0), max(2, ANCHO // 400))

    plt.figure(figsize=(13, 5))
    plt.suptitle(f"Patente - img_{nro}")
    plt.subplot(1, 3, 1)
    imshow(cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB), new_fig=False,
           color_img=True, title="1) Patente detectada")
    plt.subplot(1, 3, 2)
    imshow(patente_bin, new_fig=False, title="2) Patente binarizada")
    plt.subplot(1, 3, 3)
    imshow(cv2.cvtColor(recorte_viz, cv2.COLOR_BGR2RGB), new_fig=False,
           color_img=True, title=f"3) {len(caracteres)} caracteres")
    plt.tight_layout()
    plt.show(block=False)

plt.show(block=True)