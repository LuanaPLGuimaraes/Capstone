"""
Treina (ou retreina) o modelo YOLO de UM tipo.

COPIA deste mesmo arquivo pra dentro de final/modelos/<tipo>/ (ampola,
blister_992, blister_bloco, blister_org, blister_org2, ...) sem editar
nada — o script descobre sozinho de qual tipo ele é, lendo o nome da
PRÓPRIA pasta onde foi colocado (AQUI.name). Por isso TIPO tem que ser
exatamente igual ao nome da pasta em dataset_yolo/<tipo>/ (o que o
organizar_dataset.py gerou) — mesma convenção usada em todo o resto do
projeto (inspecionar.py, main.py).

Pra retreinar (mais fotos rotuladas, dataset mudou, etc.): roda esse
MESMO comando de novo, sem editar nada. Cada rodada cria uma subpasta
nova aqui dentro (o Ultralytics numera sozinho: treino_completo,
treino_completo2, ...) — treinos antigos não são sobrescritos, fica um
histórico.

O --project do treino é sempre resolvido pra caminho ABSOLUTO antes de
usar (Nota técnica 6.5 do registro de testes) — não cai na pasta
`runs/` errada.

Uso (rodando de dentro de final/modelos/<tipo>, ex: final/modelos/blister_992):
    python treinar.py
    python treinar.py --epochs 150
    python treinar.py --nome treino_completo_v2
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

AQUI = Path(__file__).resolve().parent               # final/modelos/<tipo> — ex: final/modelos/blister_992
TIPO = AQUI.name                                      # deduzido da própria pasta, ex: "blister_992"
DATA_YAML = (AQUI / ".." / ".." / "dataset_yolo" / TIPO / "data.yaml").resolve()
MODELO_BASE_PADRAO = (AQUI / ".." / ".." / "scripts_comuns" / "yolov8n.pt").resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_YAML), help="Caminho pro data.yaml")
    ap.add_argument("--nome", default="treino_completo", help="Nome da subpasta desse treino")
    ap.add_argument("--modelo-base", default=str(MODELO_BASE_PADRAO),
                     help="Pesos de partida — yolov8n.pt pré-treinado no COCO; "
                          "pra continuar de um treino anterior, aponta pro best.pt dele")
    ap.add_argument("--epochs", type=int, default=100, help="Épocas de treino de verdade")
    ap.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem de entrada")
    ap.add_argument("--batch", type=int, default=-1,
                     help="Tamanho do batch. -1 deixa o Ultralytics escolher pela memória disponível")
    args = ap.parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise SystemExit(
            f"Não achei {data_yaml}\n"
            f"Rodou o organizar_dataset.py pro tipo '{TIPO}' antes? É ele que gera esse data.yaml."
        )

    print(f"data.yaml:   {data_yaml}")
    print(f"Salvando em: {AQUI / args.nome}")
    print(f"Modelo base: {args.modelo_base}  |  epochs={args.epochs}  imgsz={args.imgsz}")
    print("-" * 60)

    modelo = YOLO(args.modelo_base)
    modelo.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(AQUI),   # já é absoluto, pois AQUI vem de .resolve()
        name=args.nome,
    )

    print("-" * 60)
    print(f"Treino concluído. Pesos finais em: {AQUI / args.nome / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()