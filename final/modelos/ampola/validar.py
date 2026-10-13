"""
AMPOLA — Valida o modelo treinado. Três relatórios:

1. Resumo geral (precision/recall/mAP por classe, no ponto de operação
   padrão do Ultralytics) — visão geral de como o treino foi.
2. Varredura de limiar de confiança focada na classe CRÍTICA (por
   padrão, a que tem "vazi" no nome — ampola_vazia) — pra cada limiar,
   calcula recall e precision SÓ dessa classe, e recomenda o maior
   limiar que ainda garante recall = 1,0 (nenhuma vazia escapando),
   que é a exigência de negócio (nenhuma vazia pode ser APROVADA).

   Atenção aos nomes: "falso positivo" da classe vazia = o modelo
   marcou uma ampola boa como vazia por engano (rejeição indevida,
   incômodo mas não grave). "Falso negativo" da classe vazia = uma
   vazia de verdade que o modelo NÃO marcou como vazia (ela passa
   como se tivesse produto — esse é o erro que não pode acontecer).
   É por isso que o critério de aprovação é RECALL = 1,0, não
   precision.
3. Diagnóstico das caixas de ampola_vazia que ficaram sem detecção da
   própria classe (FN), usando o mesmo casamento por classe da seção 2
   (point a point confiável — sem ambiguidade de qual caixa é qual).
   Pra cada uma, reporta SEM decidir nada: quais previsões (de
   qualquer classe) se sobrepõem geometricamente àquela caixa
   específica, com que IoU e confiança — e salva a foto inteira
   (contexto) + um recorte com zoom (pra identificar sem erro qual
   ampola é, numa bandeja com dezenas delas).

   Também reporta accuracy caixa-a-caixa (soma das duas classes),
   calculada por casamento SEPARADO por classe (mesma lógica da seção
   2, uma vez por classe) — não por atribuição cruzada entre classes,
   que é o que causava o bug acima.

Fica dentro de final/modelos/ampola/ — caminhos relativos a essa posição.

Uso (rodando de dentro de final/modelos/ampola):
    python validar.py
    python validar.py --pesos treino_completo/weights/best.pt
    python validar.py --classe-critica ampola_vazia
    python validar.py --conf-diagnostico 0.05
"""
import argparse
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

AQUI = Path(__file__).resolve().parent  # final/modelos/ampola
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def iou_xywhn(a, b):
    """IoU entre duas caixas (cx, cy, w, h) em coordenadas normalizadas
    0-1. Válido mesmo sem saber o tamanho da imagem: escalar os dois
    pelo mesmo fator (W, H) não muda a razão interseção/união."""
    ax0, ay0 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax1, ay1 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx1, by1 = b[0] + b[2] / 2, b[1] + b[3] / 2

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih

    area_a = max(0.0, a[2]) * max(0.0, a[3])
    area_b = max(0.0, b[2]) * max(0.0, b[3])
    uniao = area_a + area_b - inter
    return inter / uniao if uniao > 0 else 0.0


def ler_gt_classe(label_path: Path, classe_id: int):
    """Lê do .txt só as caixas da classe pedida (cx, cy, w, h). Só lê
    linhas cujo índice de classe no arquivo bate com classe_id — não
    tem ambiguidade nenhuma sobre qual é a classe real dessa caixa."""
    if not label_path.exists():
        return []
    boxes = []
    for linha in label_path.read_text(encoding="utf-8").splitlines():
        partes = linha.split()
        if not partes:
            continue
        if int(partes[0]) != classe_id:
            continue
        cx, cy, w, h = map(float, partes[1:5])
        boxes.append((cx, cy, w, h))
    return boxes


def avaliar_threshold(gt_por_imagem, preds_por_imagem, classe_id, conf_thresh, iou_match=0.5):
    """Agrega TP/FP/FN da classe_id, em TODAS as imagens, nesse limiar
    de confiança. Casamento é sempre DENTRO da mesma classe (só compara
    GT da classe_id com previsões da classe_id) — por isso não tem
    risco de uma caixa 'roubar' o pareamento de outra classe."""
    tp = fp = fn = 0
    for img_key in gt_por_imagem:
        gt_boxes = gt_por_imagem[img_key]
        preds = [p for p in preds_por_imagem.get(img_key, [])
                 if p["cls"] == classe_id and p["conf"] >= conf_thresh]
        preds.sort(key=lambda p: -p["conf"])

        gt_usado = [False] * len(gt_boxes)
        for p in preds:
            melhor_iou, melhor_j = 0.0, -1
            for j, gtb in enumerate(gt_boxes):
                if gt_usado[j]:
                    continue
                iou = iou_xywhn(p["box"], gtb)
                if iou > melhor_iou:
                    melhor_iou, melhor_j = iou, j
            if melhor_iou >= iou_match:
                gt_usado[melhor_j] = True
                tp += 1
            else:
                fp += 1
        fn += gt_usado.count(False)
    return tp, fp, fn


def encontrar_fn(gt_por_imagem, preds_por_imagem, classe_id, conf_thresh, iou_match=0.5):
    """Mesmo casamento do avaliar_threshold (só dentro da classe_id),
    mas devolve QUAIS caixas reais ficaram sem previsão casada (FN),
    com a imagem e a caixa — não só a contagem."""
    fn_boxes = []
    for img_key, gt_boxes in gt_por_imagem.items():
        preds = [p for p in preds_por_imagem.get(img_key, [])
                 if p["cls"] == classe_id and p["conf"] >= conf_thresh]
        preds.sort(key=lambda p: -p["conf"])

        gt_usado = [False] * len(gt_boxes)
        for p in preds:
            melhor_iou, melhor_j = 0.0, -1
            for j, gtb in enumerate(gt_boxes):
                if gt_usado[j]:
                    continue
                iou = iou_xywhn(p["box"], gtb)
                if iou > melhor_iou:
                    melhor_iou, melhor_j = iou, j
            if melhor_iou >= iou_match:
                gt_usado[melhor_j] = True
        for j, gtb in enumerate(gt_boxes):
            if not gt_usado[j]:
                fn_boxes.append((img_key, gtb))
    return fn_boxes


def overlaps_na_caixa(img_key, box_gt, preds_por_imagem, nomes, conf_floor=0.05, iou_min=0.2):
    """Pra UMA caixa real específica, lista TODAS as previsões (de
    qualquer classe, sem exclusividade — nenhuma previsão é 'reservada'
    pra essa caixa nem tirada de outra) cujo IoU com ela é >= iou_min,
    ordenadas por IoU decrescente. É só uma consulta geométrica local —
    não decide nada, só mostra o que de fato se sobrepõe a essa caixa."""
    candidatos = []
    for p in preds_por_imagem.get(img_key, []):
        if p["conf"] < conf_floor:
            continue
        iou = iou_xywhn(p["box"], box_gt)
        if iou >= iou_min:
            candidatos.append({"classe": nomes[p["cls"]], "conf": p["conf"], "iou": iou})
    candidatos.sort(key=lambda c: -c["iou"])
    return candidatos


def desenhar_erro(img_path: Path, box_norm, out_dir: Path, prefixo: str):
    """Salva DUAS imagens pra essa caixa real sem detecção:
    - '<prefixo>_contexto.jpg': a foto inteira com a caixa marcada —
      pra situar onde ela fica na bandeja.
    - '<prefixo>_zoom.jpg': um recorte ampliado só da região da caixa
      (com margem) — pra não ter dúvida de qual ampola exata é, numa
      bandeja com dezenas delas coladas."""
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    cx, cy, bw, bh = box_norm
    x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
    x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)

    out_dir.mkdir(parents=True, exist_ok=True)

    # contexto: foto inteira com a caixa marcada
    img_contexto = img.copy()
    espessura = max(2, round(w / 400))
    cv2.rectangle(img_contexto, (x0, y0), (x1, y1), (0, 0, 255), espessura)
    cv2.imwrite(str(out_dir / f"{prefixo}_contexto.jpg"), img_contexto)

    # zoom: recorte com margem de 1.5x o tamanho da caixa pra cada lado,
    # ampliado, pra identificar a ampola exata sem ambiguidade
    margem_x, margem_y = int(bw * w * 1.5), int(bh * h * 1.5)
    cx0, cy0 = max(0, x0 - margem_x), max(0, y0 - margem_y)
    cx1, cy1 = min(w, x1 + margem_x), min(h, y1 + margem_y)
    recorte = img[cy0:cy1, cx0:cx1].copy()
    if recorte.size > 0:
        fator = max(1, 400 // max(recorte.shape[0], recorte.shape[1]))
        if fator > 1:
            recorte = cv2.resize(recorte, (recorte.shape[1] * fator, recorte.shape[0] * fator),
                                  interpolation=cv2.INTER_NEAREST)
        # a caixa, nas coordenadas do recorte (já ampliado)
        rx0, ry0 = (x0 - cx0) * fator, (y0 - cy0) * fator
        rx1, ry1 = (x1 - cx0) * fator, (y1 - cy0) * fator
        cv2.rectangle(recorte, (rx0, ry0), (rx1, ry1), (0, 0, 255), 2)
        cv2.imwrite(str(out_dir / f"{prefixo}_zoom.jpg"), recorte)

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default="treino_completo/weights/best.pt",
                     help="Caminho pros pesos treinados (relativo a essa pasta)")
    ap.add_argument("--data", default=str((AQUI / ".." / ".." / "dataset_yolo" / "ampola" / "data.yaml").resolve()),
                     help="Caminho pro data.yaml")
    ap.add_argument("--classe-critica", default=None,
                     help="Nome da classe crítica a analisar no detalhe (default: a que tem 'vazi' no nome)")
    ap.add_argument("--conf-min", type=float, default=0.05)
    ap.add_argument("--conf-max", type=float, default=0.95)
    ap.add_argument("--conf-step", type=float, default=0.05)
    ap.add_argument("--conf-diagnostico", type=float, default=None,
                     help="Limiar usado no diagnóstico de FN (default: o mesmo valor de --conf-min, "
                          "o mais permissivo — é o que tem mais chance de mostrar um erro que NÃO é de limiar)")
    args = ap.parse_args()

    pesos = (AQUI / args.pesos).resolve()
    if not pesos.exists():
        raise SystemExit(f"Não achei os pesos em {pesos}")

    data_yaml = Path(args.data).resolve()
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    nomes = cfg["names"]  # {0: "ampola_vazia", 1: "ampola_com_produto"}
    dataset_root = Path(cfg["path"])
    val_rel = cfg["val"]
    val_images_dir = (dataset_root / val_rel).resolve()
    val_labels_dir = Path(str(val_images_dir).replace("images", "labels"))

    if args.classe_critica:
        classe_nome = args.classe_critica
        classe_id = [i for i, n in nomes.items() if n == classe_nome][0]
    else:
        candidatos_classe = [(i, n) for i, n in nomes.items() if "vazi" in n.lower()]
        if not candidatos_classe:
            raise SystemExit(
                "Não achei automaticamente uma classe com 'vazi' no nome. "
                "Passa --classe-critica <nome> explicitamente."
            )
        classe_id, classe_nome = candidatos_classe[0]

    print(f"Pesos:            {pesos}")
    print(f"Classe crítica:   {classe_nome} (id={classe_id})")
    print("=" * 70)

    modelo = YOLO(str(pesos))

    # --- 1. Resumo geral (ponto de operação padrão do Ultralytics) ---
    print("RESUMO GERAL (ponto de operação padrão)")
    print("-" * 70)
    # project=/name= sempre absolutos (Nota técnica 6.5) — sem isso, o
    # Ultralytics ignora o caminho relativo e salva em capstone/runs/detect/.
    pasta_val = AQUI / "validacao"
    metrics = modelo.val(data=str(data_yaml), verbose=False, project=str(AQUI), name="validacao")
    print(f"  (plots/resultados dessa validação salvos em: {pasta_val})")
    for idx, cls_id in enumerate(metrics.box.ap_class_index):
        nome = modelo.names[int(cls_id)]
        p = metrics.box.p[idx]
        r = metrics.box.r[idx]
        ap50 = metrics.box.ap50[idx]
        print(f"  {nome:25s}  precision={p:.3f}  recall={r:.3f}  mAP50={ap50:.3f}")
    print(f"  mAP50 geral: {metrics.box.map50:.3f}  |  mAP50-95 geral: {metrics.box.map:.3f}")
    print("=" * 70)

    # --- carrega GT (por classe) e previsões, uma vez só ---
    imagens = sorted(p for p in val_images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imagens:
        raise SystemExit(f"Nenhuma imagem de validação encontrada em {val_images_dir}")
    imagem_por_stem = {p.stem: p for p in imagens}

    gt_por_classe = {cid: {} for cid in nomes}
    preds_por_imagem = {}
    for img_path in imagens:
        label_path = val_labels_dir / (img_path.stem + ".txt")
        for cid in nomes:
            gt_por_classe[cid][img_path.stem] = ler_gt_classe(label_path, cid)

        resultado = modelo.predict(str(img_path), conf=0.001, verbose=False)[0]
        preds = []
        for box in resultado.boxes:
            preds.append({
                "cls": int(box.cls[0]),
                "conf": float(box.conf[0]),
                "box": tuple(box.xywhn[0].tolist()),
            })
        preds_por_imagem[img_path.stem] = preds

    gt_por_imagem = gt_por_classe[classe_id]

    # --- 2. Varredura de limiar focada na classe crítica ---
    print(f"VARREDURA DE LIMIAR — classe '{classe_nome}'")
    print("-" * 70)
    print("Atenção aos nomes: aqui, 'falso positivo' = ampola BOA marcada como vazia por engano")
    print("(rejeição indevida, incômodo). 'Falso negativo' = ampola VAZIA que o modelo deixou passar")
    print("como se tivesse produto (o erro grave). Por isso o alvo é RECALL = 1,0, não precision.")

    total_gt = sum(len(v) for v in gt_por_imagem.values())
    print(f"  {len(imagens)} imagens de validação, {total_gt} caixas reais da classe '{classe_nome}'")
    print(f"\n  {'conf':>6s}  {'recall':>8s}  {'precision':>10s}  {'TP':>4s}  {'FP':>4s}  {'FN':>4s}")

    melhor_conf_recall1 = None
    n_passos = round((args.conf_max - args.conf_min) / args.conf_step) + 1
    for i in range(n_passos):
        conf = round(args.conf_min + i * args.conf_step, 4)
        tp, fp, fn = avaliar_threshold(gt_por_imagem, preds_por_imagem, classe_id, conf)
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        marca = " <-- recall perfeito" if recall == 1.0 else ""
        print(f"  {conf:>6.2f}  {recall:>8.3f}  {precision:>10.3f}  {tp:>4d}  {fp:>4d}  {fn:>4d}{marca}")
        if recall == 1.0:
            melhor_conf_recall1 = (conf, precision)

    print("-" * 70)
    if melhor_conf_recall1:
        conf, precision = melhor_conf_recall1
        print(f"Recomendação: use conf >= {conf:.2f} na inspeção — é o limiar MAIS ALTO (menos falso")
        print(f"alarme) que ainda garante recall 1,0 na classe '{classe_nome}' nesse conjunto de")
        print(f"validação (precision nesse ponto: {precision:.3f}).")
    else:
        print(f"NENHUM limiar testado chegou a recall 1,0 pra '{classe_nome}' — o modelo ainda deixa")
        print(f"passar alguma vazia em TODOS os pontos testados. Isso não é um problema de calibração")
        print(f"de limiar — veja o diagnóstico abaixo pra entender se é confusão de classe ou falta de dado.")
    print("\n(Esse teste usa só o conjunto de VALIDAÇÃO — mesmas fotos vistas de longe durante o")
    print("treino, não é um teste cego. Pra confirmar de verdade, idealmente usar fotos novas que")
    print("nunca passaram nem por treino nem por validação.)")
    print("=" * 70)

    # --- 3. Accuracy caixa-a-caixa (casamento SEPARADO por classe, sem atribuição cruzada) ---
    conf_diag = args.conf_diagnostico if args.conf_diagnostico is not None else args.conf_min
    print(f"ACCURACY CAIXA-A-CAIXA (conf={conf_diag:.2f})")
    print("-" * 70)
    tp_soma = total_soma = 0
    for cid, nome_classe in nomes.items():
        tp_c, _, fn_c = avaliar_threshold(gt_por_classe[cid], preds_por_imagem, cid, conf_diag)
        total_c = tp_c + fn_c
        tp_soma += tp_c
        total_soma += total_c
    accuracy = tp_soma / total_soma if total_soma else float("nan")
    print(f"  Accuracy (caixas reais corretamente detectadas da própria classe / total de caixas reais): "
          f"{accuracy:.3f}")
    print(f"  ({tp_soma} corretas de {total_soma} caixas reais no total, somando as duas classes)")
    print("  (Não é a métrica padrão de detecção — não existe um universo fixo de 'negativos' numa foto —")
    print("  mas responde na prática a mesma pergunta: de tudo que existe de verdade, quanto o modelo achou certo.)")
    print("=" * 70)

    # --- 4. Diagnóstico: onde estão as caixas de ampola_vazia sem detecção (FN) ---
    print(f"DIAGNÓSTICO DOS FALSOS NEGATIVOS — classe '{classe_nome}' (conf={conf_diag:.2f})")
    print("-" * 70)
    fn_boxes = encontrar_fn(gt_por_imagem, preds_por_imagem, classe_id, conf_diag)
    if not fn_boxes:
        print(f"  Nenhum falso negativo de '{classe_nome}' nesse limiar.")
    else:
        pasta_diag = AQUI / "diagnostico_erros"
        print(f"  {len(fn_boxes)} caixa(s) real(is) de '{classe_nome}' sem detecção da própria classe.")
        print(f"  Imagens salvas em: {pasta_diag}  (arquivo _contexto = foto inteira, _zoom = recorte ampliado)")
        for i, (img_key, box) in enumerate(fn_boxes, start=1):
            img_path = imagem_por_stem.get(img_key)
            print(f"\n  [{i}] imagem: {img_path.name if img_path else img_key}")
            candidatos = overlaps_na_caixa(img_key, box, preds_por_imagem, nomes)
            if not candidatos:
                print(f"       Nenhuma previsão (de nenhuma classe) se sobrepõe a essa caixa — o modelo")
                print(f"       simplesmente não viu nada ali, não é confusão de classe.")
            else:
                print(f"       Previsões que se sobrepõem a essa caixa específica (sem exclusividade —")
                print(f"       podem incluir previsões que também pertencem a uma ampola vizinha):")
                for c in candidatos:
                    print(f"         - {c['classe']:25s}  conf={c['conf']:.3f}  IoU com essa caixa={c['iou']:.3f}")
            if img_path:
                desenhar_erro(img_path, box, pasta_diag, f"{img_key}_fn{i}")
    print("=" * 70)


if __name__ == "__main__":
    main()