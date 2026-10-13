"""
FINAL — Inspeciona uma foto de bandeja: roda o modelo YOLO do tipo
indicado, devolve a MESMA foto com caixas VERMELHAS só ao redor das
embalagens vazias, e devolve as contagens (quantas embalagens no
total, quantas vazias). É a peça que a função orquestradora (chamada
pelo LabVIEW depois que ele tira e salva a foto) usa pra apontar pro
operador quais embalagens parecem vazias.

Script genérico, igual ao organizar_dataset.py — não um por tipo como
treinar.py/validar.py, porque a lógica aqui (carrega o modelo do tipo,
desenha a classe crítica) é idêntica pra qualquer produto; só muda o
--tipo.

PACOTE DE PRODUÇÃO: pra levar isso pro computador do LabVIEW, não
precisa de organizar_dataset.py/treinar.py/validar.py nem das pastas de
dataset — aquilo é ferramenta de desenvolvimento (treinar/validar o
modelo). O `.pt` de cada tipo já carrega os nomes das classes dentro
dele (gravados no treino) — não precisa levar o data.yaml nem o
dataset_yolo/ pra isso. Basta por tipo: o arquivo `best.pt` + este
script + `ultralytics`/`opencv-python` instalados.

Uso como FUNÇÃO (pra importar na função orquestradora que o LabVIEW
chama):
    from inspecionar import inspecionar_imagem
    resultado = inspecionar_imagem("foto.jpg", tipo="ampola")
    resultado["caminho_saida"]     # imagem com as vazias marcadas
    resultado["total_detectado"]   # quantas embalagens no total
    resultado["total_vazias"]      # quantas são vazias

Uso como SCRIPT (pra testar sozinho, sem LabVIEW):
    python inspecionar.py --tipo ampola --imagem foto.jpg
    python inspecionar.py --tipo ampola --imagem foto.jpg --conf-vazia 0.10 --saida resultado.jpg

Dois limiares de confiança, propositalmente diferentes:
- --conf-vazia (default 0.05): BAIXO de propósito. Isso não é um
  sistema de aprovação automática, é um apontador visual pro operador
  — ele ainda olha a bandeja. Um alarme falso (ampola boa marcada à
  toa) custa 1 segundo de conferência a mais; deixar passar uma vazia
  sem marcar é o erro que não pode acontecer (Nota técnica 8.2 do
  registro de testes). Por isso prioriza pegar todo mundo suspeito.
- --conf-padrao (default 0.25): usado só pra CONTAR as embalagens que
  não são a classe crítica. Se usasse o limiar baixo aqui também,
  ruído de baixa confiança infla a contagem total — pra "quantas
  embalagens tem na bandeja" você quer um número confiável, não o mais
  sensível possível.

O modelo de cada tipo fica em cache em memória (_MODELOS_CACHE) depois
da primeira chamada — importante numa linha de produção chamando isso
foto após foto: não recarrega o modelo do disco toda hora.
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

AQUI = Path(__file__).resolve().parent  # final/scripts_comuns
COR_VAZIA = (0, 0, 255)  # vermelho, BGR (OpenCV usa BGR, não RGB)

_MODELOS_CACHE = {}  # {tipo: (modelo YOLO, classe_id_critica, classe_nome_critica)}


def _carregar_modelo(tipo: str, pesos: str = None):
    """Carrega (ou reaproveita do cache) o modelo YOLO de um tipo e
    identifica a classe crítica (a que tem 'vazi' no nome, mesma
    convenção do validar.py). Os nomes das classes vêm do PRÓPRIO
    arquivo .pt (modelo.names) — não precisa de data.yaml nenhum, o
    que deixa o pacote de produção bem menor (só o .pt por tipo)."""
    if tipo in _MODELOS_CACHE:
        return _MODELOS_CACHE[tipo]

    # produção: modelos/<tipo>/best.pt é SUBPASTA daqui (inspecionar.py
    # fica na raiz de producao/, não dentro de um scripts_comuns/ irmão
    # de modelos/ como em final/) — por isso sem "..": achatado, sem o
    # treino_completo/weights/ do ambiente de desenvolvimento (lá, esse
    # caminho existe pra manter histórico de vários treinos; em produção
    # só interessa O modelo atual daquele tipo)
    pesos_path = Path(pesos) if pesos else (AQUI / "modelos" / tipo / "best.pt")
    pesos_path = pesos_path.resolve()
    if not pesos_path.exists():
        raise SystemExit(f"Não achei os pesos do tipo '{tipo}' em {pesos_path}")

    modelo = YOLO(str(pesos_path))
    nomes = modelo.names  # {0: 'ampola_vazia', 1: 'ampola_com_produto'} — gravado no treino
    candidatos = [(i, n) for i, n in nomes.items() if "vazi" in n.lower()]
    if not candidatos:
        raise SystemExit(f"Não achei uma classe com 'vazi' no nome pro tipo '{tipo}' (classes: {nomes})")
    classe_id, classe_nome = candidatos[0]

    _MODELOS_CACHE[tipo] = (modelo, classe_id, classe_nome)
    return _MODELOS_CACHE[tipo]


def inspecionar_imagem(imagem_path, tipo: str, conf_vazia: float = 0.05, conf_padrao: float = 0.25,
                        saida=None, pesos: str = None):
    """Roda o modelo do `tipo` na foto. Desenha caixa VERMELHA em cada
    detecção da classe crítica (vazia) com confiança >= conf_vazia —
    não desenha nada nas outras classes, só o que precisa da atenção
    do operador. Conta: vazias (limiar conf_vazia, sensível) e as
    demais embalagens (limiar conf_padrao, mais confiável, só pra
    contagem).

    Devolve um dict:
        caminho_saida    — Path da imagem salva com as vazias marcadas
        total_detectado  — total de embalagens contadas (vazias + outras)
        total_vazias      — quantas são da classe crítica
        total_outras      — quantas são das demais classes
    """
    imagem_path = Path(imagem_path)
    modelo, classe_id, _classe_nome = _carregar_modelo(tipo, pesos)

    conf_minima = min(conf_vazia, conf_padrao)
    resultado = modelo.predict(str(imagem_path), conf=conf_minima, verbose=False)[0]
    img = cv2.imread(str(imagem_path))
    if img is None:
        raise SystemExit(f"Não consegui abrir a imagem {imagem_path}")
    h, w = img.shape[:2]

    espessura = max(2, round(w / 300))
    escala_fonte = max(0.6, w / 1000)
    n_vazias = 0
    n_outras = 0
    for box in resultado.boxes:
        cls = int(box.cls[0])
        conf_box = float(box.conf[0])
        is_vazia = (cls == classe_id)

        if is_vazia:
            if conf_box < conf_vazia:
                continue
            n_vazias += 1
            cx, cy, bw, bh = box.xywhn[0].tolist()
            x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x0, y0), (x1, y1), COR_VAZIA, espessura)
            texto = f"VAZIA {conf_box:.2f}"
            ty = max(20, y0 - 10)
            cv2.putText(img, texto, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, escala_fonte, COR_VAZIA, espessura)
        else:
            if conf_box < conf_padrao:
                continue
            n_outras += 1

    if saida is None:
        saida = imagem_path.with_name(imagem_path.stem + "_inspecionado" + imagem_path.suffix)
    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(saida), img)

    return {
        "caminho_saida": saida,
        "total_detectado": n_vazias + n_outras,
        "total_vazias": n_vazias,
        "total_outras": n_outras,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tipo", required=True, help="Tipo do produto (ampola, blister_992, blister_bloco, ...)")
    ap.add_argument("--imagem", required=True, help="Caminho da foto a inspecionar")
    ap.add_argument("--conf-vazia", type=float, default=0.05,
                     help="Confiança mínima pra marcar/contar uma vazia (default 0.05 — ver docstring do módulo)")
    ap.add_argument("--conf-padrao", type=float, default=0.25,
                     help="Confiança mínima pra contar as demais embalagens (default 0.25)")
    ap.add_argument("--saida", default=None, help="Caminho da imagem de saída (default: <imagem>_inspecionado.<ext>)")
    ap.add_argument("--pesos", default=None,
                     help="Override do caminho dos pesos (default: final/modelos/<tipo>/treino_completo/weights/best.pt)")
    args = ap.parse_args()

    r = inspecionar_imagem(
        args.imagem, args.tipo, conf_vazia=args.conf_vazia, conf_padrao=args.conf_padrao,
        saida=args.saida, pesos=args.pesos,
    )
    print(f"Detectei {r['total_detectado']} embalagem(ns) nessa foto, {r['total_vazias']} vazia(s) "
          f"(tipo '{args.tipo}').")
    print(f"Imagem salva em: {r['caminho_saida']}")


if __name__ == "__main__":
    main()