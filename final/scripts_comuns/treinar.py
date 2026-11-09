"""
Treina (ou retreina) o modelo YOLO de UM tipo.

Script único para todos os tipos: --tipo tem que ser exatamente igual
ao nome da pasta em final/dataset_yolo/<tipo>/ (o que o
organizar_dataset.py gerou) — mesma convenção usada em todo o resto do
projeto (inspecionar.py, main.py). Os resultados ficam em
final/modelos/<tipo>/.

Pra retreinar (mais fotos rotuladas, dataset mudou, etc.): roda esse
MESMO comando de novo, sem editar nada. Cada rodada cria uma subpasta
nova (o Ultralytics numera sozinho: treino_completo, treino_completo2,
...) — treinos antigos não são sobrescritos, fica um histórico.

O --project do treino é sempre resolvido pra caminho ABSOLUTO antes de
usar (Nota técnica 6.5 do registro de testes) — não cai na pasta
`runs/` errada.

Uso (a partir da raiz do repositório):
    python final/scripts_comuns/treinar.py --tipo blister_992
    python final/scripts_comuns/treinar.py --tipo ampola --epochs 150
    python final/scripts_comuns/treinar.py --tipo ampola --nome treino_completo_v2
    python final/scripts_comuns/treinar.py --tipo ampola --device cpu   # força CPU mesmo tendo GPU

GPU: usa a placa NVIDIA automaticamente se o PyTorch do venv tiver CUDA
(confere com: python -c "import torch; print(torch.cuda.is_available())").
Se aparecer "CPU" no cabeçalho, o torch instalado é a versão só-CPU.
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

AQUI = Path(__file__).resolve().parent               # final/scripts_comuns
PASTA_FINAL = AQUI.parent                            # final/
PASTA_MODELOS = PASTA_FINAL / "modelos"
PASTA_DATASET_YOLO = PASTA_FINAL / "dataset_yolo"
MODELO_BASE_PADRAO = AQUI / "yolov8n.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tipo", required=True, help="Tipo do produto (ampola, blister_992, blister_bloco, ...)")
    ap.add_argument("--data", default=None,
                     help="Caminho pro data.yaml (default: final/dataset_yolo/<tipo>/data.yaml)")
    ap.add_argument("--nome", default="treino_completo", help="Nome da subpasta desse treino")
    ap.add_argument("--modelo-base", default=str(MODELO_BASE_PADRAO),
                     help="Pesos de partida — yolov8n.pt pré-treinado no COCO; "
                          "pra continuar de um treino anterior, aponta pro best.pt dele")
    ap.add_argument("--epochs", type=int, default=100, help="Épocas de treino de verdade")
    ap.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem de entrada")
    ap.add_argument("--batch", type=int, default=-1,
                     help="Tamanho do batch. -1 deixa o Ultralytics escolher pela memória disponível")
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu",
                     help="'0' = primeira GPU NVIDIA, 'cpu' = processador. Padrão: GPU se houver")
    args = ap.parse_args()

    pasta_tipo = PASTA_MODELOS / args.tipo
    data_yaml = Path(args.data or PASTA_DATASET_YOLO / args.tipo / "data.yaml").resolve()
    if not data_yaml.exists():
        raise SystemExit(
            f"Não achei {data_yaml}\n"
            f"Rodou o organizar_dataset.py pro tipo '{args.tipo}' antes? É ele que gera esse data.yaml."
        )

    print(f"data.yaml:   {data_yaml}")
    print(f"Salvando em: {pasta_tipo / args.nome}")
    print(f"Modelo base: {args.modelo_base}  |  epochs={args.epochs}  imgsz={args.imgsz}")
    if args.device == "cpu":
        print("Dispositivo: CPU (mais lento — veja a nota sobre GPU no topo do arquivo)")
    else:
        print(f"Dispositivo: GPU {args.device} ({torch.cuda.get_device_name(int(args.device))})")
    print("-" * 60)

    modelo = YOLO(args.modelo_base)
    modelo.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(pasta_tipo),   # já é absoluto: PASTA_FINAL vem de .resolve()
        name=args.nome,
    )

    print("-" * 60)
    print(f"Treino concluído. Pesos finais em: {pasta_tipo / args.nome / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
