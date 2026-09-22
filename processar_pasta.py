"""
PASSO 2.5 — Rodar detectar_ampolas.py em varias fotos de uma vez

Uso:
    python processar_pasta.py --input dataset/ampolas/ --out resultados_ampolas/ --intensidade-media-max 27
"""
import argparse
from pathlib import Path

import cv2

from detectar_tampas import detectar_tampas
from detectar_ampolas import agrupar_por_linha, medir_pitch, expandir_para_ampola, desenhar

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Pasta com as fotos")
    ap.add_argument("--out", default="resultados_ampolas", help="Pasta de saida")
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
    ap.add_argument("--intensidade-media-max", type=int, default=255,
                     help="0-255. Media de escuridao do blob tem que ser <= isso pra contar como tampa")
    ap.add_argument("--ignorar-borda", action="store_true")
    ap.add_argument("--tolerancia-y", type=int, default=25)
    ap.add_argument("--pitch-min", type=int, default=100)
    ap.add_argument("--pitch-max", type=int, default=350)
    ap.add_argument("--pitch-direcao", choices=["esquerda", "direita"], default="esquerda",
                     help="Pra que lado o corpo da ampola se estende a partir da tampa. Confira na sua foto!")
    args = ap.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    imagens = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imagens:
        raise SystemExit(f"Nenhuma imagem encontrada em {input_dir}")

    print(f"{'arquivo':40s} {'tampas':>8s} {'ampolas':>8s}")
    print("-" * 60)
    contagens = []
    for img_path in imagens:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"{img_path.name:40s} {'ERRO AO ABRIR':>8s}")
            continue

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
            # nao deu pra medir o espacamento da grade nessa foto (ex: tampas
            # demais faltando, ou parametros nao calibrados pra ela) 
            print(f"{img_path.name:40s} {len(boxes_tampa):>8d} {'SEM PITCH':>8s}")
            vis = img.copy()
            for (x1, y1, x2, y2) in boxes_tampa:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.imwrite(str(out_dir / img_path.name), vis)
            continue

        boxes_ampola = [expandir_para_ampola(b, pitch, args.pitch_direcao) for b in boxes_tampa]
        vis = desenhar(img, boxes_tampa, boxes_ampola)
        cv2.imwrite(str(out_dir / img_path.name), vis)

        print(f"{img_path.name:40s} {len(boxes_tampa):>8d} {len(boxes_ampola):>8d}")
        contagens.append(len(boxes_ampola))

    if contagens:
        print("-" * 60)
        print(f"Media: {sum(contagens)/len(contagens):.1f}  "
              f"Min: {min(contagens)}  Max: {max(contagens)}")
        print("\nSe a contagem variar muito de foto pra foto (ou aparecer 'SEM")
        print("PITCH'), e sinal de que os parametros nao generalizam bem pra")
        print(f"aquela foto - abra ela em {out_dir}/ e compare com o original.")


if __name__ == "__main__":
    main()