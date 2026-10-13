"""
FINAL — Organizar o dataset já rotulado (foto + .txt revisados no
LabelImg) em treino/validação, na estrutura de pastas que o YOLO
(ultralytics) espera pra treinar.

Lê pares (imagem, .txt) válidos de uma pasta de origem, embaralha com
uma semente FIXA, separa em treino/validação numa proporção
configurável, e COPIA (não move — a pasta original fica intacta) pra:

    <out>/images/train/*.jpg
    <out>/images/val/*.jpg
    <out>/labels/train/*.txt
    <out>/labels/val/*.txt
    <out>/data.yaml           <- arquivo de config que o YOLO lê pra treinar

Se achar imagem sem .txt (ainda não revisada) ou .txt sem imagem
(par quebrado), avisa e NÃO inclui esse arquivo no dataset — você
decide se quer revisar essa foto depois e rodar de novo.

Classes: por padrão lê direto do `classes.txt` que já está dentro da
pasta --input (o mesmo arquivo que o LabelImg usa) — não precisa
digitar os nomes de novo, e evita o risco de digitar em ordem
diferente do que foi rotulado (os índices nos .txt são posicionais,
então ordem errada = classe errada, sem erro nenhum na hora). Só passe
--classes se quiser sobrescrever isso por algum motivo.

Uso (rodando de dentro de final/scripts_comuns):
    python organizar_dataset.py --input ../dataset/DATASET_ampola --out ../dataset_yolo/ampola --val-frac 0.2
    python organizar_dataset.py --input ../dataset/DATASET_Blister-Bloco --out ../dataset_yolo/blister_bloco

    # só se precisar forçar nomes diferentes do classes.txt:
    python organizar_dataset.py --input ../dataset/DATASET_ampola --out ../dataset_yolo/ampola --classes ampola_vazia,ampola_com_produto
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


def ler_classes(input_dir: Path):
    """Lê os nomes de classe direto do classes.txt dentro da pasta de
    input (o mesmo arquivo que o LabelImg usa) — uma classe por linha,
    na ordem dos índices usados nos .txt de rótulo."""
    classes_path = input_dir / "classes.txt"
    if not classes_path.exists():
        return None
    linhas = [linha.strip() for linha in classes_path.read_text(encoding="utf-8").splitlines()]
    return [linha for linha in linhas if linha]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Pasta com as fotos + .txt já revisados no LabelImg")
    ap.add_argument("--out", default="dataset_yolo", help="Pasta de saída (estrutura YOLO)")
    ap.add_argument("--classes", default=None,
                     help="Nomes das classes, separados por vírgula, na ordem dos índices usados nos .txt "
                          "(ex: ampola_vazia,ampola_com_produto). Opcional — por padrão lê do classes.txt "
                          "que já está dentro de --input. Só use isso pra sobrescrever.")
    ap.add_argument("--val-frac", type=float, default=0.2, help="Fração pra validação (0.2 = 20%%)")
    ap.add_argument("--seed", type=int, default=42, help="Semente do embaralhamento (fixa = reprodutível)")
    args = ap.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out)

    if args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
        origem_classes = "--classes (passado manualmente)"
    else:
        classes = ler_classes(input_dir)
        origem_classes = f"{input_dir / 'classes.txt'}"
        if not classes:
            raise SystemExit(
                f"Não achei (ou está vazio) {input_dir / 'classes.txt'}.\n"
                f"Ou cria esse arquivo (mesmo que o LabelImg usa pra rotular), ou passa "
                f"--classes nome1,nome2 manualmente."
            )

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

    print(f"Classes (de {origem_classes}): {', '.join(classes)}")
    print(f"Total de pares válidos: {len(pares)}")
    print(f"Treino: {len(train_pares)}  |  Validação: {len(val_pares)}")
    print(f"Estrutura criada em: {out_dir.resolve()}")
    print(f"Config do YOLO salva em: {data_yaml}")


if __name__ == "__main__":
    main()