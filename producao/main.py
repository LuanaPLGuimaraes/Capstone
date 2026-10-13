"""
PRODUÇÃO — ponto de entrada que o LabVIEW vai chamar (hoje, via linha
de comando pra teste; quando a integração de verdade for definida, me
fala como o LabVIEW vai chamar isso — comando externo, Python Node,
etc. — que a gente ajusta).

Recebe --tipo (que produto é) e --imagem (a foto que acabou de ser
tirada e salva), chama o inspecionar.py do tipo certo, e imprime o
resultado em UMA LINHA de JSON no stdout — formato fácil de qualquer
outra ferramenta (LabVIEW incluso, via suas VIs de JSON) ler e
processar, sem precisar fazer parsing de texto solto.

Uso:
    python main.py --tipo ampola --imagem teste.png
"""
import argparse
import json
from pathlib import Path

from inspecionar import inspecionar_imagem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tipo", required=True, help="Tipo do produto (ampola, blister_992, blister_bloco, ...)")
    ap.add_argument("--imagem", required=True, help="Caminho da foto a inspecionar")
    ap.add_argument("--conf-vazia", type=float, default=0.05)
    ap.add_argument("--conf-padrao", type=float, default=0.25)
    args = ap.parse_args()

    resultado = inspecionar_imagem(
        args.imagem, args.tipo, conf_vazia=args.conf_vazia, conf_padrao=args.conf_padrao,
    )

    saida = {
        "tipo": args.tipo,
        "caminho_imagem_entrada": str(Path(args.imagem).resolve()),
        "caminho_imagem_saida": str(resultado["caminho_saida"]),
        "total_detectado": resultado["total_detectado"],
        "total_vazias": resultado["total_vazias"],
        "total_outras": resultado["total_outras"],
    }
    print(json.dumps(saida, ensure_ascii=False))


if __name__ == "__main__":
    main()