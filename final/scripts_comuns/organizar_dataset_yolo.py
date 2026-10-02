"""
FINAL — Organizar o dataset já rotulado (foto + .txt revisados no
LabelImg) em treino/validação, na estrutura de pastas que o YOLO
(ultralytics) espera pra treinar.

Le pares (imagem, .txt) válidos de uma pasta de origem, embaralha com
uma semente FIXA, separa emtreino/validação numa proporção configurável, 
e COPIA (não move — a pasta original fica intacta) pra:

    <out>/images/train/*.jpg
    <out>/images/val/*.jpg
    <out>/labels/train/*.txt
    <out>/labels/val/*.txt
    <out>/data.yaml           <- arquivo de config que o YOLO lê pra treinar

Se achar imagem sem .txt (ainda não revisada) ou .txt sem imagem
(par quebrado), avisa e NÃO inclui esse arquivo no dataset — vocêw
decide se quer revisar essa foto depois e rodar de novo.

Uso exemplo:
    python organizar_dataset_yolo.py --input ../ampola/dataset --out ../ampola/dataset_yolo --classes ampola_vazia,ampola_com_produto --val-frac 0.2
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def encontrar_pares(input_dir: Path):
    """Retorna lista de (imagem, label) pra cada par válido, avisando
    sobre pares quebrados (imagem sem rótulo, ou rótulo sem imagem)."""
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
    ap.add_argument("--input", required=True, help="Pasta com as fotos + .txt já revisados no LabelImg")
    ap.add_argument("--out", default="dataset_yolo", help="Pasta de saída (estrutura YOLO)")
    ap.add_argument("--classes", required=True,
                     help="Nomes das classes na ordem dos índices usados nos .txt, separados por vírgula "
                          "(ex: ampola_vazia,ampola_com_produto). Tem que bater com o classes.txt que você usou no LabelImg.")
    ap.add_argument("--val-frac", type=float, default=0.2, help="Fração pra validação (0.2 = 20%%)")
    ap.add_argument("--seed", type=int, default=42, help="Semente do embaralhamento (fixa = reprodutível)")
    args = ap.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes:
        raise SystemExit("--classes vazio — passe pelo menos um nome de classe.")

    pares = encontrar_pares(input_dir)
    if not pares:
        raise SystemExit("Nenhum par (imagem + rótulo) válido encontrado.")

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
    linhas_classes = "\n".join(f"  {i}: {nome}" for i, nome in enumerate(classes))
    data_yaml.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{linhas_classes}\n",
        encoding="utf-8",
    )

    print(f"Classes: {', '.join(classes)}")
    print(f"Total de pares válidos: {len(pares)}")
    print(f"Treino: {len(train_pares)}  |  Validação: {len(val_pares)}")
    print(f"Estrutura criada em: {out_dir.resolve()}")
    print(f"Config do YOLO salva em: {data_yaml}")


if __name__ == "__main__":
    main()