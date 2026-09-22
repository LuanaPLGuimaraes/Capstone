"""
FASE YOLO — passo 1.5: pre-rotular as fotos usando o pipeline classico,
CONFERIR e CORRIGIR no LabelImg em vez de desenhar ~70
caixas por foto do zero.

O que esse script faz, por foto:
  1. Roda o mesmo pipeline classico de sempre (detectar tampa -> agrupar
     por linha -> medir pitch -> expandir pra ampola inteira).
  2. Converte cada caixa (pixels) pro formato de rotulo do YOLO (fracao
     0-1 da imagem).
  3. Salva um arquivo <nome_da_foto>.txt do LADO da foto original, com
     uma linha por ampola: "<classe> <centro_x> <centro_y> <largura> <altura>".
  4. Toda caixa nasce marcada com a classe --classe-padrao (por padrao,
     0 = ampola_vazia) — porque o pipeline classico nao sabe distinguir
     vazia de com produto, so localiza. 

Uso:
    python gerar_rotulos_yolo.py --input dataset/fotos --intensidade-media-max 27
"""
import argparse
from pathlib import Path

import cv2

from detectar_tampas import detectar_tampas
from detectar_ampolas import agrupar_por_linha, medir_pitch, expandir_para_ampola

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ordem das classes = o numero que cada uma vira no arquivo .txt.
# Se mudar aqui, tem que ser a MESMA ordem que voce configurar no LabelImg.
CLASSES = ["ampola_vazia", "ampola_com_produto"]


def caixa_para_yolo(box, largura_img, altura_img):
    """(x1,y1,x2,y2) em pixels -> (centro_x, centro_y, largura, altura)
    em fracao 0-1 da imagem, que e o que o formato YOLO espera."""
    x1, y1, x2, y2 = box
    # trava a caixa dentro dos limites da imagem 
    x1 = max(0, min(x1, largura_img))
    x2 = max(0, min(x2, largura_img))
    y1 = max(0, min(y1, altura_img))
    y2 = max(0, min(y2, altura_img))

    cx = (x1 + x2) / 2 / largura_img
    cy = (y1 + y2) / 2 / altura_img
    w = (x2 - x1) / largura_img
    h = (y2 - y1) / altura_img
    return cx, cy, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Pasta com as fotos a rotular")
    ap.add_argument("--classe-padrao", type=int, default=0,
                     help=f"Indice da classe que toda caixa nasce marcada (0={CLASSES[0]}, 1={CLASSES[1]})")
    ap.add_argument("--sobrescrever", action="store_true",
                     help="Regera o .txt mesmo se ja existir (CUIDADO: perde correcoes manuais ja feitas)")
    # mesmos parametros calibraveis do pipeline classico 
    ap.add_argument("--dark-threshold", type=int, default=45)
    ap.add_argument("--area-min", type=int, default=500)
    ap.add_argument("--area-max", type=int, default=2300)
    ap.add_argument("--min-distance-picos", type=int, default=12)
    ap.add_argument("--sobreposicao-min", type=float, default=0.25)
    ap.add_argument("--margem-juncao", type=int, default=6)
    ap.add_argument("--altura-min", type=int, default=0)
    ap.add_argument("--altura-max", type=int, default=10_000)
    ap.add_argument("--largura-min", type=int, default=0)
    ap.add_argument("--largura-max", type=int, default=10_000)
    ap.add_argument("--preenchimento-min", type=float, default=0.0)
    ap.add_argument("--intensidade-media-max", type=int, default=255)
    ap.add_argument("--ignorar-borda", action="store_true")
    ap.add_argument("--tolerancia-y", type=int, default=25)
    ap.add_argument("--pitch-min", type=int, default=100)
    ap.add_argument("--pitch-max", type=int, default=350)
    ap.add_argument("--pitch-direcao", choices=["esquerda", "direita"], default="esquerda")
    args = ap.parse_args()

    input_dir = Path(args.input)
    imagens = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imagens:
        raise SystemExit(f"Nenhuma imagem encontrada em {input_dir}")

    # classes.txt: o LabelImg le esse arquivo pra saber o nome e a ordem
    # das classes (a ordem tem que bater com o numero salvo no .txt)
    classes_path = input_dir / "classes.txt"
    if not classes_path.exists() or args.sobrescrever:
        classes_path.write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
        print(f"classes.txt escrito em {classes_path}")

    print(f"{'arquivo':40s} {'ampolas':>8s} {'status':>12s}")
    print("-" * 65)

    gerados = 0
    pulados_existente = 0
    pulados_sem_pitch = 0

    for img_path in imagens:
        label_path = img_path.with_suffix(".txt")

        if label_path.exists() and not args.sobrescrever:
            print(f"{img_path.name:40s} {'':>8s} {'ja existe':>12s}")
            pulados_existente += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"{img_path.name:40s} {'':>8s} {'ERRO AO ABRIR':>12s}")
            continue
        altura_img, largura_img = img.shape[:2]

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

        try:
            linhas = agrupar_por_linha(boxes_tampa, tolerancia_y=args.tolerancia_y)
            pitch = medir_pitch(linhas, pitch_min=args.pitch_min, pitch_max=args.pitch_max)
        except ValueError:
            # nao deu pra medir a grade nessa foto — nao escreve .txt,
            # essa foto fica pra voce rotular manualmente do zero
            print(f"{img_path.name:40s} {len(boxes_tampa):>8d} {'SEM PITCH':>12s}")
            pulados_sem_pitch += 1
            continue

        boxes_ampola = [expandir_para_ampola(b, pitch, args.pitch_direcao) for b in boxes_tampa]

        linhas_txt = []
        for box in boxes_ampola:
            cx, cy, w, h = caixa_para_yolo(box, largura_img, altura_img)
            if w <= 0 or h <= 0:
                continue  # caixa degenerada (ex: colada na borda) — ignora
            linhas_txt.append(f"{args.classe_padrao} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        label_path.write_text("\n".join(linhas_txt) + "\n", encoding="utf-8")
        print(f"{img_path.name:40s} {len(linhas_txt):>8d} {'gerado':>12s}")
        gerados += 1

    print("-" * 65)
    print(f"Gerados: {gerados}  |  Ja existiam (pulados): {pulados_existente}  |  Sem pitch (pulados): {pulados_sem_pitch}")
    print("\nAbra a pasta no LabelImg (formato YOLO) — as caixas ja vao")
    print("aparecer desenhadas. Confira posicao/tamanho e marque a classe")
    print("certa (vazia/com produto) em cada uma.")


if __name__ == "__main__":
    main()