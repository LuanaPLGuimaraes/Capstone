"""
PASSO 1 — Detectar e contar ampolas na bandeja usando visao computacional
CLASSICA (sem deep learning, sem OCR).

Importante para o TCC: isso NAO e OCR. OCR (Tesseract etc.) serve para ler
TEXTO em uma imagem — nao ha texto nas ampolas, entao OCR nao se aplica
aqui. O que estamos fazendo e "visao computacional classica": tecnicas de
processamento de imagem baseadas em regras (limiarizacao, morfologia,
watershed), sem nenhum modelo treinado nem dataset rotulado. E a etapa
antes de qualquer classificador.

IDEIA (por que funciona nesta foto):
  Cada ampola tem uma tampa/anel PRETO numa das pontas, bem mais escura
  que o vidro e o fundo branco da bandeja. Isso da um contraste alto e
  facil de isolar so com limiar de intensidade.

Etapas do algoritmo (cada uma vira uma imagem de debug, ver --debug):
  1. Escala de cinza -> limiar (threshold) isola pixels bem escuros
     (as tampas, mas tambem cantos escuros do fundo e ruido).
  2. Remove blobs grandes demais (fundo fora da bandeja) e ruido pequeno
     (abertura morfologica).
  3. Tampas vizinhas costumam se tocar na imagem e virar 1 blob so —
     watershed (transformada de distancia + picos locais) separa as
     encostadas antes de contar.
  4. Filtra por area esperada de uma tampa, pra descartar tanto ruido
     pequeno (graos do produto tambem sao escuros) quanto blobs grandes.
  5. Conta os blobs finais = numero de ampolas detectadas.

Uso:
    python detectar_tampas.py --image foto.jpg --out resultado.png --debug

Com --debug, tambem salva (prefixo = mesmo nome do --out):
    <out>_1_mascara_bruta.png     -> direto do threshold, antes de limpar
    <out>_2_mascara_limpa.png     -> depois de remover blobs de fundo
    <out>_3_watershed.png         -> cada blob final com uma cor diferente

Use essas imagens pra decidir o que ajustar (guia completo no README.md).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def _iou(box_a, box_b):
    """Mede o quanto duas caixas se sobrepoem, de 0 (nao se tocam) a 1
    (caixas identicas). E a conta padrao: area da intersecao dividida
    pela area da uniao."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def _gap(box_a, box_b):
    """Distancia entre 2 caixas em cada eixo. 0 ou negativo = ja se
    tocam/sobrepoem naquele eixo; positivo = folga real entre elas."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    gap_x = max(ax1, bx1) - min(ax2, bx2)
    gap_y = max(ay1, by1) - min(ay2, by2)
    return gap_x, gap_y


def mesclar_caixas_proximas(boxes, areas, sobreposicao_min=0.25, margem_juncao=6, area_min_fragmento=500):
    """Junta caixas sobrepostas OU apenas grudadas (pouca folga entre
    elas) numa unica caixa maior. `areas` e a area (em pixels) de cada
    caixa em `boxes`, na mesma ordem.

    Duas regras BEM diferentes, de propósito:
    1. Sobreposicao forte (IoU >= sobreposicao_min): sempre junta. E o
       caso "a mesma tampa foi detectada 2x, quase uma em cima da
       outra" — nunca e o caso de 2 tampas DIFERENTES (2 tampas reais
       nunca ocupam o mesmo lugar).
    2. So GRUDADAS, sem sobrepor (folga <= margem_juncao): so junta se
       PELO MENOS UMA das duas caixas for pequena demais pra ser uma
       tampa sozinha (area < area_min_fragmento). Isso importa: tampas
       reais e vizinhas costumam ENCOSTAR uma na outra na foto (por
       isso o watershed existe, pra separar quem se toca) — se
       juntassemos qualquer par grudado, desfariamos essa separacao e
       contariamos 2 tampas reais como 1. So faz sentido juntar por
       "estar grudado" quando um dos pedacos e pequeno demais pra ser
       uma tampa inteira sozinho (sinal de que e um FRAGMENTO de uma
       tampa que foi cortada ao meio, nao uma tampa vizinha de verdade).
    """
    itens = [{"box": b, "area": a} for b, a in zip(boxes, areas)]
    mudou = True
    while mudou:
        mudou = False
        for i in range(len(itens)):
            for j in range(i + 1, len(itens)):
                a, b = itens[i], itens[j]
                junta_por_sobreposicao = _iou(a["box"], b["box"]) >= sobreposicao_min
                gap_x, gap_y = _gap(a["box"], b["box"])
                grudadas = gap_x <= margem_juncao and gap_y <= margem_juncao
                tem_fragmento = a["area"] < area_min_fragmento or b["area"] < area_min_fragmento
                junta_por_fragmento = grudadas and tem_fragmento
                if junta_por_sobreposicao or junta_por_fragmento:
                    ax1, ay1, ax2, ay2 = a["box"]
                    bx1, by1, bx2, by2 = b["box"]
                    unido = (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))
                    novo = {"box": unido, "area": a["area"] + b["area"]}
                    itens = [it for k, it in enumerate(itens) if k != i and k != j]
                    itens.append(novo)
                    mudou = True
                    break
            if mudou:
                break
    return [it["box"] for it in itens]


def detectar_tampas(
    img: np.ndarray,
    dark_threshold: int = 45,
    area_min: int = 500,
    area_max: int = 2300,
    min_distance_picos: int = 12,
    area_max_blob_fundo: int = 5000,
    sobreposicao_min: float = 0.25,
    margem_juncao: int = 6,
    altura_min: int = 0,
    altura_max: int = 10_000,
    largura_min: int = 0,
    largura_max: int = 10_000,
    preenchimento_min: float = 0.0,
    intensidade_media_max: int = 255,
    ignorar_borda: bool = False,
):
    """Retorna (boxes, debug) — boxes = lista de (x1,y1,x2,y2), uma por
    tampa detectada; debug = dict com as imagens intermediarias, uteis
    pra entender/ajustar os parametros.

    Os parametros altura_min/altura_max, largura_min/largura_max,
    preenchimento_min, intensidade_media_max e ignorar_borda sao filtros
    EXTRAS (alem da area) pra descartar blobs que passaram no limiar de
    escuridao mas nao tem cara de tampa de verdade — por exemplo, um
    amontoado de graos escuros do produto. Por padrao eles vem
    "desligados" (bem permissivos) pra nao mudar nada em quem ja estava
    funcionando; ajuste-os olhando o --debug quando o limiar sozinho
    nao for suficiente."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) pixels bem escuros = candidatos a tampa (e tambem fundo escuro/ruido)
    _, mask_bruta = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask_bruta, cv2.MORPH_OPEN, kernel, iterations=1)

    # 2) descarta blobs enormes (fundo escuro fora da bandeja, cantos da imagem)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    mask_limpa = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < area_max_blob_fundo:
            mask_limpa[labels == i] = 255

    # 3) separa tampas encostadas (watershed sobre a transformada de distancia)
    dist = ndi.distance_transform_edt(mask_limpa)
    coords = peak_local_max(dist, min_distance=min_distance_picos, labels=mask_limpa)
    peak_mask = np.zeros(dist.shape, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)
    ws = watershed(-dist, markers, mask=mask_limpa)

    # 4) monta uma caixa por blob do watershed, SEM filtrar nada ainda —
    # guarda tambem a area de cada uma (precisa pra decidir no passo 5 se
    # e um fragmento pequeno ou uma tampa "de tamanho normal")
    caixas_cruas = []
    areas_cruas = []
    for lbl in range(1, ws.max() + 1):
        ys, xs = np.where(ws == lbl)
        if len(xs) == 0:
            continue
        caixas_cruas.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
        areas_cruas.append(len(xs))

    # 5) junta pedacos grudados/sobrepostos ANTES de filtrar por area.
    # Importante fazer nessa ordem: uma tampa real pode ser cortada em 2
    # pedacos pequenos que, sozinhos, cada um nao passaria no filtro de
    # area — so o pedaco JUNTO tem o tamanho certo de uma tampa inteira.
    # Se filtrassemos por area antes de juntar, perderiamos a tampa
    # inteira (os 2 pedacos pequenos seriam descartados um por um).
    # (Duas tampas DIFERENTES que soh encostam uma na outra nao sao
    # juntadas aqui — ver a explicacao dentro de mesclar_caixas_proximas.)
    caixas_unidas = mesclar_caixas_proximas(
        caixas_cruas,
        areas_cruas,
        sobreposicao_min=sobreposicao_min,
        margem_juncao=margem_juncao,
        area_min_fragmento=area_min,
    )

    # 6) SO AGORA filtra por area/formato/preenchimento/intensidade,
    # recalculando essas medidas em cima da caixa ja unida.
    h_img, w_img = gray.shape
    boxes = []
    for (x1, y1, x2, y2) in caixas_unidas:
        largura, altura = (x2 - x1 + 1), (y2 - y1 + 1)

        # pixels que sao realmente "escuros" (mascara limpa) dentro dessa
        # caixa — usamos isso em vez de recontar do watershed pq a caixa
        # pode ser a uniao de 2 pedacos que eram labels diferentes
        regiao_mask = mask_limpa[y1 : y2 + 1, x1 : x2 + 1] > 0
        area = int(regiao_mask.sum())
        if area < area_min or area > area_max:
            continue

        # filtro de FORMATO: uma tampa de verdade tem altura e largura
        # numa faixa esperada. Um amontoado de graos de produto pode ter
        # a mesma AREA de uma tampa, mas um formato bem diferente (mais
        # comprido, mais fino, mais torto) — a area sozinha nao pega isso.
        if not (altura_min <= altura <= altura_max):
            continue
        if not (largura_min <= largura <= largura_max):
            continue

        # filtro de PREENCHIMENTO: uma tampa e um blobzao solido — o
        # retangulo ao redor dela vem quase todo preenchido de pixel
        # escuro. Textura de produto forma um blob cheio de buracos/
        # espalhado, que preenche so uma fracao pequena do retangulo.
        preenchimento = area / float(largura * altura)
        if preenchimento < preenchimento_min:
            continue

        # filtro de INTENSIDADE MEDIA: em vez de so "passou no limiar",
        # confere se a MEDIA de escuridao dos pixels escuros dentro da
        # caixa e realmente baixa. Sombra/reflexo/produto as vezes cruza
        # o limiar mas na media nao e tao escuro quanto uma tampa real.
        regiao_gray = gray[y1 : y2 + 1, x1 : x2 + 1]
        intensidade_media = float(regiao_gray[regiao_mask].mean()) if area > 0 else 255.0
        if intensidade_media > intensidade_media_max:
            continue

        # descarta blob colado na borda da foto (corte de camera/bandeja,
        # quase nunca e uma tampa real e completa)
        if ignorar_borda and (x1 == 0 or y1 == 0 or x2 == w_img - 1 or y2 == h_img - 1):
            continue

        boxes.append((x1, y1, x2, y2))

    # imagem colorida so pra visualizar os labels do watershed (debug)
    ws_color = np.zeros((*ws.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    cores = rng.integers(50, 255, size=(ws.max() + 1, 3))
    for lbl in range(1, ws.max() + 1):
        ws_color[ws == lbl] = cores[lbl]

    debug = {
        "mascara_bruta": mask_bruta,
        "mascara_limpa": mask_limpa,
        "watershed": ws_color,
    }
    return boxes, debug


def desenhar_resultado(img: np.ndarray, boxes):
    vis = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(vis, str(i), (cx - 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Foto da bandeja de ampolas")
    ap.add_argument("--out", default="resultado.png", help="Imagem de saida com as caixas desenhadas")
    ap.add_argument("--dark-threshold", type=int, default=45,
                     help="0-255. Mais BAIXO = mais exigente (so o que for MUITO escuro vira tampa)")
    ap.add_argument("--area-min", type=int, default=500, help="Area minima (px) pra contar como tampa")
    ap.add_argument("--area-max", type=int, default=2300, help="Area maxima (px) pra contar como tampa")
    ap.add_argument("--min-distance-picos", type=int, default=12,
                     help="Distancia minima (px) entre 2 tampas pro watershed separar. Mais BAIXO = separa mais (risco de dividir 1 tampa em 2)")
    ap.add_argument("--sobreposicao-min", type=float, default=0.25,
                     help="0 a 1. Se duas caixas se sobrepoem mais que isso, viram 1 so (corrige tampa cortada em 2 pedacos). Diminuir junta mais; aumentar junta menos")
    ap.add_argument("--margem-juncao", type=int, default=6,
                     help="Pixels de folga: 2 caixas grudadas (sem sobrepor) com folga <= isso viram 1 so (corrige tampa cortada em pedacos vizinhos, ex: coluna da borda). Aumentar junta mais; 0 desliga")
    ap.add_argument("--altura-min", type=int, default=0, help="Filtro de formato: altura minima (px) de uma tampa")
    ap.add_argument("--altura-max", type=int, default=10_000, help="Filtro de formato: altura maxima (px) de uma tampa")
    ap.add_argument("--largura-min", type=int, default=0, help="Filtro de formato: largura minima (px) de uma tampa")
    ap.add_argument("--largura-max", type=int, default=10_000, help="Filtro de formato: largura maxima (px) de uma tampa")
    ap.add_argument("--preenchimento-min", type=float, default=0.0,
                     help="0 a 1. Fracao minima do retangulo que precisa estar preenchida de pixel escuro. Textura de produto costuma preencher pouco (~0.2-0.4); tampa solida preenche bastante (~0.6+)")
    ap.add_argument("--intensidade-media-max", type=int, default=255,
                     help="0-255. Media de escuridao dos pixels do blob tem que ser <= isso pra contar como tampa")
    ap.add_argument("--ignorar-borda", action="store_true", help="Descarta blobs colados na borda da foto")
    ap.add_argument("--debug", action="store_true", help="Salva as imagens intermediarias pra ajudar a calibrar")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Nao consegui abrir a imagem: {args.image}")

    boxes, debug = detectar_tampas(
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

    vis = desenhar_resultado(img, boxes)
    cv2.imwrite(args.out, vis)

    if args.debug:
        out_stem = Path(args.out).with_suffix("")
        cv2.imwrite(f"{out_stem}_1_mascara_bruta.png", debug["mascara_bruta"])
        cv2.imwrite(f"{out_stem}_2_mascara_limpa.png", debug["mascara_limpa"])
        cv2.imwrite(f"{out_stem}_3_watershed.png", debug["watershed"])
        print(f"Imagens de debug salvas com prefixo: {out_stem}_*.png")

    print(f"Ampolas detectadas: {len(boxes)}")
    print(f"Resultado salvo em: {args.out}")
    print("Confira visualmente se a contagem bate com a foto original.")


if __name__ == "__main__":
    main()