"""
PRODUÇÃO — Inspeciona uma foto de bandeja: roda o modelo YOLO do tipo
indicado, devolve a MESMA foto com caixas VERMELHAS só ao redor das
embalagens vazias, e devolve as contagens (quantas embalagens no
total, quantas vazias). É a peça que a função orquestradora (chamada
pelo LabVIEW depois que ele tira e salva a foto) usa pra apontar pro
operador quais embalagens parecem vazias.

Script genérico: a lógica (carregar o modelo do tipo, desenhar a
classe crítica) é idêntica pra qualquer produto; só muda o --tipo.

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
import numpy as np
from ultralytics import YOLO

AQUI = Path(__file__).resolve().parent  # producao/
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


def _rodar_e_marcar(img_bgr, modelo, classe_id, conf_vazia, conf_padrao):
    """Núcleo compartilhado: roda o modelo numa imagem já carregada em
    memória (array BGR, formato que o OpenCV/YOLO esperam) e desenha as
    caixas vermelhas nas vazias, direto no array `img_bgr` (em vez de
    conferir se veio de um arquivo ou de uma câmera via LabVIEW — essa
    diferença fica por conta de quem chama esta função).

    Usado tanto por inspecionar_imagem (entrada = caminho de arquivo)
    quanto por inspecionar_array (entrada = array do LabVIEW, sem
    nenhum arquivo .jpg no meio). Devolve (n_vazias, n_outras)."""
    h, w = img_bgr.shape[:2]
    conf_minima = min(conf_vazia, conf_padrao)
    # Ultralytics aceita um array numpy diretamente como "source" — não
    # precisa salvar em disco pra rodar o modelo.
    resultado = modelo.predict(img_bgr, conf=conf_minima, verbose=False)[0]

    espessura = max(2, round(w / 300))
    escala_fonte = max(0.4, w / 1500)
    espessura_texto = max(1, espessura - 1)  # texto um pouco mais fino que a borda da caixa
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
            cv2.rectangle(img_bgr, (x0, y0), (x1, y1), COR_VAZIA, espessura)
            texto = f"{conf_box:.2f}"  # só o número de confiança — sem a palavra "VAZIA"
            ty = max(20, y0 - 10)
            cv2.putText(img_bgr, texto, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, escala_fonte, COR_VAZIA, espessura_texto)
        else:
            if conf_box < conf_padrao:
                continue
            n_outras += 1

    return n_vazias, n_outras


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

    img = cv2.imread(str(imagem_path))
    if img is None:
        raise SystemExit(f"Não consegui abrir a imagem {imagem_path}")

    n_vazias, n_outras = _rodar_e_marcar(img, modelo, classe_id, conf_vazia, conf_padrao)

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


def inspecionar_array(array, tipo: str, altura: int, largura: int, canais: int = 3,
                       ordem_cor: str = "RGB", conf_vazia: float = 0.05, conf_padrao: float = 0.25,
                       pesos: str = None):
    """Igual a inspecionar_imagem, mas SEM NENHUM ARQUIVO no meio — pra
    usar com o LabVIEW Python Node, recebendo a foto como array (ex.:
    saída de "IMAQ ImageToArray"/"IMAQ ImageToRGB") em vez de um
    caminho de .jpg. Não precisa do bloco "Write JPEG File" nem de
    "Read JPEG File": a imagem entra como array e sai como array.

    Parâmetros:
        array     — a imagem vinda do LabVIEW. Pode ser uma lista
                    aninhada (do jeito que o Python Node entrega um
                    array 2D/3D do LabVIEW) ou já um array numpy.
        altura, largura — dimensões da imagem (o LabVIEW manda isso
                    separado, junto com a imagem — não dá pra adivinhar
                    só olhando a lista).
        canais    — 1 pra imagem em escala de cinza (câmera
                    monocromática), 3 pra imagem colorida.
        ordem_cor — "RGB" (padrão do IMAQ) ou "BGR". Importa só se
                    canais=3: o OpenCV/YOLO foram treinados esperando
                    BGR, então se o array vier em RGB a função inverte
                    os canais antes de rodar o modelo (e inverte de
                    volta antes de devolver, pra sair na MESMA ordem
                    que entrou — quem recebe de volta no LabVIEW não
                    precisa se preocupar com isso).

    Devolve um dict:
        imagem_saida     — a imagem (já com as caixas vermelhas) como
                            lista aninhada, nas MESMAS dimensões/ordem
                            de cor de entrada — dá pra converter de
                            volta pra Image no LabVIEW (ex.: "IMAQ
                            ArrayToImage"/"IMAQ ArrayColorImage") e
                            mostrar direto, sem salvar em disco.
        total_detectado, total_vazias, total_outras — iguais a
                            inspecionar_imagem.
    """
    modelo, classe_id, _classe_nome = _carregar_modelo(tipo, pesos)

    img = np.array(array, dtype=np.uint8)
    forma_esperada = (altura, largura) if canais == 1 else (altura, largura, canais)
    img = img.reshape(forma_esperada)

    if canais == 1:
        # escala de cinza -> precisa de 3 canais pra desenhar a caixa
        # vermelha (um pixel "cinza" não tem como ficar vermelho).
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif ordem_cor.upper() == "RGB":
        img_bgr = img[:, :, ::-1].copy()  # RGB -> BGR
    else:
        img_bgr = img.copy()

    n_vazias, n_outras = _rodar_e_marcar(img_bgr, modelo, classe_id, conf_vazia, conf_padrao)

    # devolve na mesma ordem de cor que entrou, pra quem recebe no
    # LabVIEW não ter que se preocupar com a inversão BGR/RGB
    if canais == 3 and ordem_cor.upper() == "RGB":
        img_saida = img_bgr[:, :, ::-1]
    else:
        img_saida = img_bgr

    return {
        "imagem_saida": img_saida.tolist(),
        "total_detectado": n_vazias + n_outras,
        "total_vazias": n_vazias,
        "total_outras": n_outras,
    }


def inspecionar_array_planos(plano_r, plano_g, plano_b, tipo: str, altura: int, largura: int,
                              conf_vazia: float = 0.05, conf_padrao: float = 0.25, pesos: str = None):
    """Igual a inspecionar_array, mas pensada pra casar direto com a
    saída de "IMAQ ColorImageToArray" do LabVIEW, que devolve a imagem
    colorida como 3 arrays 2D SEPARADOS (um plano por canal R, G, B)
    em vez de um array 3D único — assim não precisa montar/intercalar
    os 3 planos em um array só do lado do LabVIEW antes de chamar o
    Python Node; manda os 3 soltos (cada um na sua entrada do node)
    que a função já junta aqui dentro.

    Devolve os 3 planos de volta, já com as caixas vermelhas
    desenhadas, prontos pra remontar a imagem no LabVIEW com a função
    inversa (procura "ArrayToColorImage" na busca da paleta).
    """
    modelo, classe_id, _classe_nome = _carregar_modelo(tipo, pesos)

    r = np.array(plano_r, dtype=np.uint8).reshape((altura, largura))
    g = np.array(plano_g, dtype=np.uint8).reshape((altura, largura))
    b = np.array(plano_b, dtype=np.uint8).reshape((altura, largura))
    img_rgb = np.dstack((r, g, b))
    img_bgr = img_rgb[:, :, ::-1].copy()  # RGB -> BGR, pro modelo

    n_vazias, n_outras = _rodar_e_marcar(img_bgr, modelo, classe_id, conf_vazia, conf_padrao)

    img_rgb_saida = img_bgr[:, :, ::-1]  # BGR -> RGB, de volta pro LabVIEW
    return {
        "plano_r_saida": img_rgb_saida[:, :, 0].tolist(),
        "plano_g_saida": img_rgb_saida[:, :, 1].tolist(),
        "plano_b_saida": img_rgb_saida[:, :, 2].tolist(),
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
                     help="Override do caminho dos pesos (default: modelos/<tipo>/best.pt)")
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