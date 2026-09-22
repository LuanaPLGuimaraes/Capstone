"""
validacao_classico.py

Mini script de validacao da abordagem classica (deteccao da ampola inteira,
via detectar_ampolas.py) contra o gabarito ja anotado (rotulos YOLO) em
dataset/ampolas.

So valida LOCALIZACAO/CONTAGEM (pareia caixa detectada com caixa do
gabarito por IoU), nao classificacao presenca/ausencia de ampola.

Rode este arquivo na raiz do projeto (mesma pasta de detectar_tampas.py
e detectar_ampolas.py), com: python validacao_classico.py
"""

import os
import glob
from pathlib import Path

import cv2

from detectar_tampas import detectar_tampas
from detectar_ampolas import agrupar_por_linha, medir_pitch, expandir_para_ampola

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
PASTA_DATASET = "dataset/ampolas"   # jpg e txt (formato YOLO) juntos nessa pasta
IOU_THRESHOLD = 0.5

PARAMS_TAMPA = dict(
    dark_threshold=45,
    area_min=500,
    area_max=2300,
    min_distance_picos=12,
    sobreposicao_min=0.25,
    margem_juncao=6,
    altura_min=0,
    altura_max=10_000,
    largura_min=0,
    largura_max=10_000,
    preenchimento_min=0.0,
    intensidade_media_max=27,   
    ignorar_borda=False,
)
TOLERANCIA_Y = 25
PITCH_MIN, PITCH_MAX = 100, 350
PITCH_DIRECAO = "esquerda"   


def ler_rotulos_yolo(caminho_txt, largura_img, altura_img):
    """Le um rotulo YOLO (classe x_center y_center largura altura, normalizados)
    e devolve caixas em pixels (x_min, y_min, x_max, y_max)."""
    caixas = []
    if not os.path.exists(caminho_txt):
        return caixas
    with open(caminho_txt) as f:
        for linha in f:
            partes = linha.strip().split()
            if len(partes) != 5:
                continue
            _, xc, yc, w, h = map(float, partes)
            xc, yc, w, h = xc * largura_img, yc * altura_img, w * largura_img, h * altura_img
            x_min, y_min = xc - w / 2, yc - h / 2
            x_max, y_max = xc + w / 2, yc + h / 2
            caixas.append((x_min, y_min, x_max, y_max))
    return caixas


def iou(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    uniao = area_a + area_b - inter
    return inter / uniao if uniao > 0 else 0.0


def parear(deteccoes, gabarito, limiar=IOU_THRESHOLD):
    """Pareia cada deteccao com no maximo um gabarito (e vice-versa), pelo maior IoU."""
    pares = []
    usados = set()
    for i, det in enumerate(deteccoes):
        melhor_iou, melhor_j = 0.0, None
        for j, gt in enumerate(gabarito):
            if j in usados:
                continue
            v = iou(det, gt)
            if v > melhor_iou:
                melhor_iou, melhor_j = v, j
        if melhor_j is not None and melhor_iou >= limiar:
            pares.append((i, melhor_j, melhor_iou))
            usados.add(melhor_j)
    return pares


def rodar_pipeline_classico(caminho_img):
    """Reproduz o fluxo do main() de detectar_ampolas.py: detecta tampas,
    agrupa por linha, mede o pitch e expande cada tampa pra ampola inteira."""
    img = cv2.imread(caminho_img)
    if img is None:
        return []
    boxes_tampa, _ = detectar_tampas(img, **PARAMS_TAMPA)
    if not boxes_tampa:
        return []
    linhas = agrupar_por_linha(boxes_tampa, tolerancia_y=TOLERANCIA_Y)
    try:
        pitch = medir_pitch(linhas, pitch_min=PITCH_MIN, pitch_max=PITCH_MAX)
    except ValueError:
        # nao conseguiu medir a grade nessa imagem (poucas tampas/linha) -- conta 0
        return []
    return [expandir_para_ampola(b, pitch, PITCH_DIRECAO) for b in boxes_tampa]


def main():
    caminhos = sorted(glob.glob(os.path.join(PASTA_DATASET, "*.jpg")))
    if not caminhos:
        print(f"Nenhuma imagem .jpg encontrada em {PASTA_DATASET} -- confira o caminho.")
        return

    total_gt = total_det = total_vp = 0
    imagens_sem_grade = 0

    for caminho_img in caminhos:
        nome = Path(caminho_img).stem
        caminho_txt = os.path.join(PASTA_DATASET, nome + ".txt")

        img = cv2.imread(caminho_img)
        if img is None:
            continue
        altura_img, largura_img = img.shape[:2]
        gabarito = ler_rotulos_yolo(caminho_txt, largura_img, altura_img)

        deteccoes = rodar_pipeline_classico(caminho_img)
        if not deteccoes and gabarito:
            imagens_sem_grade += 1

        pares = parear(deteccoes, gabarito)

        total_gt += len(gabarito)
        total_det += len(deteccoes)
        total_vp += len(pares)

        print(f"{nome}: gabarito={len(gabarito)} detectado={len(deteccoes)} pareado={len(pares)}")

    fp = total_det - total_vp
    fn = total_gt - total_vp
    precisao = total_vp / total_det if total_det else 0
    recall = total_vp / total_gt if total_gt else 0

    print(f"\n=== RESULTADO AGREGADO (localizacao, IoU >= {IOU_THRESHOLD:.2f}) ===")
    print(f"Imagens processadas: {len(caminhos)} (sem grade detectavel: {imagens_sem_grade})")
    print(f"Total de ampolas no gabarito: {total_gt}")
    print(f"Total de ampolas detectadas:  {total_det}")
    print(f"Verdadeiros positivos (VP):   {total_vp}")
    print(f"Falsos positivos (FP):        {fp}  (deteccoes extras/duplicadas)")
    print(f"Falsos negativos (FN):        {fn}  (ampolas nao detectadas)")
    print(f"Precisao: {precisao:.3f}")
    print(f"Recall:   {recall:.3f}")


if __name__ == "__main__":
    main()