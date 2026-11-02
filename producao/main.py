r"""
PRODUÇÃO — ponto de entrada que o LabVIEW chama via System Exec.vi.

INTERFACE ACORDADA COM O LABVIEW (combinada com o time, não inventada
aqui sozinho — se mudar de novo, atualiza os dois lados juntos):

Entrada: UM ÚNICO argumento posicional — o caminho completo da foto já
salva pelo LabVIEW. Nada de --tipo, nada de --imagem nomeado.
    python main.py "C:\caminho\Fotos\Ampola\Ampola_3.png"

O TIPO não é passado — é DEDUZIDO do nome da pasta que contém a foto
(a pasta logo acima do arquivo, em minúsculo). Isso só funciona porque
o LabVIEW já salva cada foto dentro de uma pasta com o nome do produto
(ex: Fotos\Ampola\Ampola_3.png, Fotos\Blister_992\...) — ou seja, o
nome dessa pasta (em minúsculo) tem que ser EXATAMENTE igual ao nome
da pasta em modelos/<tipo>/ (ampola, blister_992, blister_bloco, ...).
Se um dia essa convenção de pastas mudar do lado do LabVIEW, essa
dedução quebra — não é um acoplamento implícito por acidente, é uma
decisão consciente pra não precisar mexer na chamada do LabVIEW toda
vez que mudar de tipo.

Saída: UMA ÚNICA LINHA no stdout, com um ";" depois de CADA um dos 3
resultados (inclusive o último) — não é mais só uma palavra:

    <total_detectado>;<total_vazias>;<status>;

    total_detectado  -> quantas embalagens foram contabilizadas ao
                         todo nessa foto (vazias + com produto).
    total_vazias     -> quantas dessas são vazias (reprovadas).
    status           -> "OK" (nenhuma vazia) ou "ERRADO" (tem pelo
                         menos uma vazia, OU deu erro no meio do
                         caminho — foto não abriu, modelo não achado,
                         etc.; erro também vira ERRADO, nunca fica em
                         silêncio. Mesmo raciocínio da Nota técnica
                         8.2: na dúvida, aponta.)

    Exemplos:
        8;0;OK;        -> 8 embalagens contadas, nenhuma vazia
        8;2;ERRADO;    -> 8 embalagens contadas, 2 vazias
        0;0;ERRADO;    -> deu erro antes de conseguir contar — os 2
                          números ficam zerados, mas a linha SEMPRE
                          sai com os 3 campos (e os 3 ";"), pro
                          parsing do LabVIEW nunca quebrar por falta
                          de campo.

Do lado do LabVIEW: lê o "standard output" do System Exec.vi e separa
por ";" — ex.: "Spreadsheet String To Array" com delimitador ";" (dá
pra usar o padrão 1D de string). Como tem um ";" depois do ÚLTIMO
campo também, o split gera um 4º elemento vazio no fim (índice 3) —
é só ignorar ele, só usa os índices 0, 1 e 2. Os índices 0 e 1 viram
número com "Decimal String To Number"; o índice 2 já é o texto
OK/ERRADO direto.

A imagem anotada (as vazias marcadas de vermelho) é salva numa pasta
IRMÃ da pasta de entrada, com "_inspecionada" no nome da PASTA (não
do arquivo) — ex: a foto veio de "Fotos\Ampola\Ampola_3.png", a
anotada vai pra "Fotos\Ampola_inspecionada\Ampola_3.png" (mesmo nome
de arquivo, só muda de pasta). Essa pasta é criada automaticamente se
ainda não existir. O LabVIEW só precisa ir atrás desse arquivo (nessa
pasta irmã, mesmo nome) depois de ler o OK/ERRADO.

Uso:
    python main.py "C:\caminho\Fotos\Ampola\Ampola_3.png"
"""
import sys
from pathlib import Path

from inspecionar import inspecionar_imagem


def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py <caminho da imagem>", file=sys.stderr)
        print("0;0;ERRADO;")
        return

    caminho_imagem = Path(sys.argv[1])
    pasta_tipo = caminho_imagem.parent              # ex: Fotos\Ampola
    tipo = pasta_tipo.name.lower()                   # ex: "ampola" (deduzido daqui)
    pasta_inspecionada = pasta_tipo.parent / f"{pasta_tipo.name}_inspecionada"  # ex: Fotos\Ampola_inspecionada
    caminho_saida = pasta_inspecionada / caminho_imagem.name  # mesmo nome de arquivo, pasta diferente

    try:
        resultado = inspecionar_imagem(caminho_imagem, tipo, saida=caminho_saida)
    except (Exception, SystemExit) as erro:
        # qualquer falha (foto não abre, modelo do tipo não encontrado,
        # pasta errada, etc.) vira ERRADO no stdout — o LabVIEW nunca
        # fica esperando uma resposta que não vem. Detalhe do erro vai
        # pro stderr, só pra debug manual, não pro que o LabVIEW lê.
        print(f"Falha ao inspecionar '{caminho_imagem}' (tipo deduzido: '{tipo}'): {erro}", file=sys.stderr)
        print("0;0;ERRADO;")
        return

    status = "ERRADO" if resultado["total_vazias"] > 0 else "OK"
    print(f"{resultado['total_detectado']};{resultado['total_vazias']};{status};")


if __name__ == "__main__":
    main()