"""
FASE 3: (TESTE 1 YOLO) Organizar o dataset ja rotulado (foto + .txt revisados no
LabelImg) em treino/validacao, na estrutura de pastas que o YOLO
(ultralytics) espera pra treinar.

Le pares (imagem, .txt) validos de uma pasta de origem, separa em
treino/validacao numa proporcao configuravel, e COPIA pra:

    <out>/images/train/*.jpg
    <out>/images/val/*.jpg
    <out>/labels/train/*.txt
    <out>/labels/val/*.txt
    <out>/data.yaml           <- arquivo de config que o YOLO le pra treinar

Se achar imagem sem .txt (ainda nao revisada) ou .txt sem imagem
(par quebrado), avisa e NAO inclui esse arquivo no dataset.

Uso:
    python organizar_dataset_yolo.py --input dataset\\ampolas --out dataset_yolo --val-frac 0.2
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASSES = ["ampola_vazia", "ampola_com_produto"]


def encontrar_pares(input_dir: Path):
    """Retorna lista de (imagem, label) pra cada par valido, avisando
    sobre pares quebrados (imagem sem rotulo, ou rotulo sem imagem)."""
    imagens = {p.stem: p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    labels = {p.stem: p for p in input_dir.glob("*.txt") if p.stem != "classes"}

    pares = []
    for stem, img_path in sorted(imagens.items()):
        label_path = labels.get(stem)
        if label_path is None:
            print(f"AVISO: {img_path.name} nao tem .txt (nao revisada?) — nao entra no dataset")
            continue
        pares.append((img_path, label_path))

    for stem, label_path in sorted(labels.items()):
        if stem not in imagens:
            print(f"AVISO: {label_path.name} nao tem imagem correspondente — ignorado")

    return pares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Pasta com as fotos + .txt ja revisados no LabelImg")
    ap.add_argument("--out", default="dataset_yolo", help="Pasta de saida (estrutura YOLO)")
    ap.add_argument("--val-frac", type=float, default=0.2, help="Fracao pra validacao (0.2 = 20%%)")
    ap.add_argument("--seed", type=int, default=42, help="Semente do embaralhamento (fixa = reprodutivel)")
    args = ap.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out)

    pares = encontrar_pares(input_dir)
    if not pares:
        raise SystemExit("Nenhum par (imagem + rotulo) valido encontrado.")

    random.Random(args.seed).shuffle(pares)
    n_val = max(1, round(len(pares) * args.val_frac))
    val_pares = pares[:n_val]
    train_pares = pares[n_val:]

    for split_nome, split_pares in [("train", train_pares), ("val", val_pares)]:
        (out_dir / "images" / split_nome).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split_nome).mkdir(parents=True, exist_ok=True)
        for img_path, label_path in split_pares:
            shutil.copy2(img_path, out_dir / "images" / split_nome / img_path.name)
            shutil.copy2(label_path, out_dir / "labels" / split_nome / label_path.name)

    data_yaml = out_dir / "data.yaml"
    linhas_classes = "\n".join(f"  {i}: {nome}" for i, nome in enumerate(CLASSES))
    data_yaml.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{linhas_classes}\n",
        encoding="utf-8",
    )

    print(f"Total de pares validos: {len(pares)}")
    print(f"Treino: {len(train_pares)}  |  Validacao: {len(val_pares)}")
    print(f"Estrutura criada em: {out_dir.resolve()}")
    print(f"Config do YOLO salva em: {data_yaml}")


if __name__ == "__main__":
    main()