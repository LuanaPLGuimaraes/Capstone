"""
PASSO 2 — Detectar a AMPOLA INTEIRA (nao so a tampa), usando a grade.

Por que nao detectamos o vidro direto: tentamos achar o contorno inteiro
da ampola por bordas (Canny) e separar cada uma sozinha. Nao funcionou
bem, como as ampolas ficam encostadas umas nas outras, o "limite" entre
duas vizinhas nao fecha direito, e a segmentacao junta ampola com ampola
ou corta uma ampola em pedacos por causa da textura do produto. Registrar
isso importa: e uma tentativa que nao deu certo, e o motivo (objetos
encostados sem borda solida entre eles) e uma limitacao conhecida de
segmentar so por contorno.

O que funciona: usar a tampa (que ja detectamos com confianca no passo 1)
como ANCORA, e a regularidade da grade pra saber o tamanho da ampola.

IDEIA:
  1. Detecta as tampas (reusa detectar_tampas.py).
  2. Agrupa tampas por LINHA (mesmo Y aproximado).
  3. Dentro de cada linha, mede a distancia entre tampas vizinhas. Como as
     ampolas ficam encostadas, essa distancia ~= comprimento de 1 ampola
     inteira. Isso e o "espacamento da grade" (pitch).
  4. Cada tampa detectada vira uma caixa que comeca nela e se estende
     `pitch` pixels na direcao do corpo da ampola (nesta foto, pra
     esquerda — confira visualmente na sua, pode ser o contrario).

Isso e valido pro CENARIO 1 (ampolas deitadas, mesma orientacao, mesmo
que faltando algumas — grade incompleta). Nao serve pro cenario de
ampolas espalhadas/giradas, isso fica pra depois.

Uso:
    python detectar_ampolas.py --image foto.jpg --out resultado_ampolas.png --pitch-direcao esquerda
"""
import argparse

import cv2
import numpy as np

from detectar_tampas import detectar_tampas


def agrupar_por_linha(boxes, tolerancia_y=25):
    """Agrupa caixas de tampa que estao na mesma linha (Y parecido).
    Retorna uma lista de linhas, cada linha e uma lista de caixas
    ordenadas da esquerda pra direita."""
    centros = []
    for (x1, y1, x2, y2) in boxes:
        cy = (y1 + y2) / 2
        centros.append((cy, (x1, y1, x2, y2)))
    centros.sort(key=lambda c: c[0])

    linhas = []
    usados = [False] * len(centros)
    for i, (cy, box) in enumerate(centros):
        if usados[i]:
            continue
        linha = [box]
        usados[i] = True
        for j in range(i + 1, len(centros)):
            if usados[j]:
                continue
            if abs(centros[j][0] - cy) < tolerancia_y:
                linha.append(centros[j][1])
                usados[j] = True
        linha.sort(key=lambda b: b[0])
        linhas.append(linha)
    return linhas


def medir_pitch(linhas, pitch_min=100, pitch_max=350):
    """Mede a distancia entre tampas vizinhas na mesma linha (o
    espacamento da grade / comprimento de 1 ampola). Ignora distancias
    fora da faixa esperada (pitch_min/pitch_max) — muito perto costuma
    ser a mesma tampa detectada em duas partes (nao separou direito no
    passo 1); muito longe costuma ser um "buraco" (tampa que faltou),
    nao o espacamento real.

    Ajuste pitch_min/pitch_max pra sua foto se a mediana parecer estranha.
    """
    distancias = []
    for linha in linhas:
        for a, b in zip(linha, linha[1:]):
            ax = (a[0] + a[2]) / 2
            bx = (b[0] + b[2]) / 2
            d = bx - ax
            if pitch_min < d < pitch_max:
                distancias.append(d)
    if not distancias:
        raise ValueError(
            "Nao consegui medir o espacamento da grade — nenhuma distancia "
            "entre tampas caiu na faixa esperada. Ajuste --pitch-min/--pitch-max."
        )
    return float(np.median(distancias))


def expandir_para_ampola(box_tampa, pitch, direcao="esquerda"):
    """Estende a caixa da tampa por `pitch` pixels na direcao do corpo
    da ampola, retornando a caixa da ampola inteira."""
    x1, y1, x2, y2 = box_tampa
    if direcao == "esquerda":
        return (int(x2 - pitch), y1, x2, y2)
    else:
        return (x1, y1, int(x1 + pitch), y2)


def desenhar(img, boxes_tampa, boxes_ampola):
    vis = img.copy()
    for box in boxes_ampola:
        cv2.rectangle(vis, box[:2], box[2:], (0, 255, 255), 2)
    for box in boxes_tampa:
        cv2.rectangle(vis, box[:2], box[2:], (0, 255, 0), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="resultado_ampolas.png")
    ap.add_argument("--dark-threshold", type=int, default=45)
    ap.add_argument("--area-min", type=int, default=500)
    ap.add_argument("--area-max", type=int, default=2300)
    ap.add_argument("--min-distance-picos", type=int, default=12)
    ap.add_argument("--sobreposicao-min", type=float, default=0.25,
                     help="0 a 1. Junta tampas detectadas em dobro (sobrepostas) numa so, antes de expandir pra ampola")
    ap.add_argument("--margem-juncao", type=int, default=6,
                     help="Pixels de folga pra juntar tampas grudadas (sem sobrepor) numa so")
    ap.add_argument("--altura-min", type=int, default=0)
    ap.add_argument("--altura-max", type=int, default=10_000)
    ap.add_argument("--largura-min", type=int, default=0)
    ap.add_argument("--largura-max", type=int, default=10_000)
    ap.add_argument("--preenchimento-min", type=float, default=0.0,
                     help="0 a 1. Fracao minima do retangulo preenchida de pixel escuro pra contar como tampa")
    ap.add_argument("--intensidade-media-max", type=int, default=255,
                     help="0-255. Media de escuridao do blob tem que ser <= isso pra contar como tampa")
    ap.add_argument("--ignorar-borda", action="store_true", help="Descarta blobs colados na borda da foto")
    ap.add_argument("--tolerancia-y", type=int, default=25,
                     help="Diferenca maxima de Y (px) pra duas tampas serem consideradas da mesma linha")
    ap.add_argument("--pitch-min", type=int, default=100)
    ap.add_argument("--pitch-max", type=int, default=350)
    ap.add_argument("--pitch-direcao", choices=["esquerda", "direita"], default="esquerda",
                     help="Pra que lado o corpo da ampola se estende a partir da tampa. Confira na sua foto!")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Nao consegui abrir a imagem: {args.image}")

    boxes_tampa, _ = detectar_tampas(
        img,
        dark_threshold=args.dark_threshold,
        area_min=args.area_min,
        area_max=args.area_max,
        min_distance_picos=args.min_distance_picos,
        sobreposicao_min=args.sobreposicao_min,
        margem_juncao=args.margem_juncao,
        altura_min=args.altura_min,
        altura_max=args.altura_max,
        largura_min=args.largura_min,
        largura_max=args.largura_max,
        preenchimento_min=args.preenchimento_min,
        intensidade_media_max=args.intensidade_media_max,
        ignorar_borda=args.ignorar_borda,
    )

    linhas = agrupar_por_linha(boxes_tampa, tolerancia_y=args.tolerancia_y)
    pitch = medir_pitch(linhas, pitch_min=args.pitch_min, pitch_max=args.pitch_max)
    print(f"Espacamento da grade (comprimento estimado de 1 ampola): {pitch:.1f} px")

    boxes_ampola = [expandir_para_ampola(b, pitch, args.pitch_direcao) for b in boxes_tampa]

    vis = desenhar(img, boxes_tampa, boxes_ampola)
    cv2.imwrite(args.out, vis)

    print(f"Tampas detectadas: {len(boxes_tampa)}")
    print(f"Ampolas (caixa cheia) desenhadas: {len(boxes_ampola)}")
    print(f"Resultado salvo em: {args.out}")
    print("Amarelo = ampola inteira, verde fino = so a tampa (conferencia).")


if __name__ == "__main__":
    main()