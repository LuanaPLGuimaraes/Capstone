"""
FINAL — Pré-rotular fotos novas usando um modelo YOLO já treinado
(em vez do pipeline clássico, que só existia pra ampola).

Generaliza a ideia do gerar_rotulos_yolo.py da fase intermediária: em vez
de gerar a caixa inicial com detecção clássica (tampa + grade), roda o
PRÓPRIO MODELO YOLO já treinado nas fotos novas, e salva as detecções
já em formato de rótulo 

Uso:
    python pre_rotular_com_modelo.py --modelo ../ampola/modelos/treino_completo/weights/best.pt --input ../ampola/dataset/fotos_iluminacao_nova --conf 0.25

Validar depois com labelImg:
cd ../final/scripts_comuns
labelImg ../dataset/DATASET_ampola ../dataset/DATASET_ampola/classes.txt

labelImg ../dataset/DATASET_Blister-org2 ../dataset/DATASET_Blister-org2/classes.txt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True,
                     help="Caminho pro .pt treinado (ex: runs/detect/treino_completo/weights/best.pt)")
    ap.add_argument("--input", required=True, help="Pasta com as fotos novas a pré-rotular")
    ap.add_argument("--conf", type=float, default=0.25,
                     help="Confiança mínima pra uma detecção virar rótulo sugerido. Mais BAIXO = sugere "
                          "mais caixas (inclusive erradas) pra você revisar; mais ALTO = sugere menos, "
                          "porém mais confiáveis (risco de faltar ampola real na sugestão)")
    ap.add_argument("--sobrescrever", action="store_true",
                     help="Regera o .txt mesmo se já existir (CUIDADO: perde correções manuais já feitas)")
    args = ap.parse_args()

    input_dir = Path(args.input)
    imagens = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imagens:
        raise SystemExit(f"Nenhuma imagem encontrada em {input_dir}")

    modelo = YOLO(args.modelo)

    # classes.txt: o LabelImg lê esse arquivo pra saber nome/ordem das
    # classes. Usa os nomes que o próprio modelo aprendeu (na ordem do
    # índice interno dele), então já bate automaticamente com o índice
    # salvo em cada linha do .txt gerado abaixo.
    nomes_ordenados = [modelo.names[i] for i in range(len(modelo.names))]
    classes_path = input_dir / "classes.txt"
    if not classes_path.exists() or args.sobrescrever:
        classes_path.write_text("\n".join(nomes_ordenados) + "\n", encoding="utf-8")
        print(f"classes.txt escrito em {classes_path} ({', '.join(nomes_ordenados)})")

    print(f"{'arquivo':40s} {'caixas':>8s} {'status':>12s}")
    print("-" * 65)

    gerados = pulados = 0
    for img_path in imagens:
        label_path = img_path.with_suffix(".txt")
        if label_path.exists() and not args.sobrescrever:
            print(f"{img_path.name:40s} {'':>8s} {'ja existe':>12s}")
            pulados += 1
            continue

        resultado = modelo.predict(str(img_path), conf=args.conf, verbose=False)[0]

        linhas_txt = []
        for box in resultado.boxes:
            cls_idx = int(box.cls[0])
            cx, cy, w, h = box.xywhn[0].tolist()  # já normalizado 0-1, formato YOLO
            linhas_txt.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        conteudo = ("\n".join(linhas_txt) + "\n") if linhas_txt else ""
        label_path.write_text(conteudo, encoding="utf-8")
        print(f"{img_path.name:40s} {len(linhas_txt):>8d} {'gerado':>12s}")
        gerados += 1

    print("-" * 65)
    print(f"Gerados: {gerados}  |  Já existiam (pulados): {pulados}")
    print(f"\nConfiança mínima usada: {args.conf} — se sugeriu poucas caixas, tente baixar")
    print("(ex: 0.15); se sugeriu caixa demais errada, suba.")
    print("Abra a pasta no LabelImg (formato YOLO) e confira TODAS as caixas — um")
    print("modelo preliminar erra classe e às vezes confunde produto com tampa/vazio.")


if __name__ == "__main__":
    main()