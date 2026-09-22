# Detecção de ampolas: visão computacional clássica + YOLO

Projeto de Capstone (fase preliminar): detectar/contar ampolas numa bandeja e classificar
presença/ausência de produto, a partir de fotos com câmera e ângulo
fixos. Duas abordagens foram desenvolvidas em sequência: primeiro visão
computacional clássica (sem treino), depois um modelo treinado (YOLO),
motivado pelas limitações encontradas na primeira.

## Conceitos

**Visão clássica não "aprende" com dados.** Os parâmetros são ajustados
manualmente olhando o resultado — não existe fase de treinamento. As 58
fotos do dataset servem, nessa etapa, para testar se os parâmetros
generalizam entre fotos, não para treinar nada.

## Estrutura do repositório

| Arquivo | Etapa | O que faz |
|---|---|---|
| `detectar_tampas.py` | Clássico — Passo 1 | Detecta a tampa/anel escuro de cada ampola (limiar + watershed) |
| `detectar_ampolas.py` | Clássico — Passo 2 | Usa a tampa como âncora e o espaçamento da grade (pitch) pra desenhar a caixa da ampola inteira |
| `processar_pasta.py` | Clássico — lote | Roda o pipeline completo (tampa + ampola inteira) em todas as fotos de uma pasta |
| `validacao_classico.py` | Clássico — validação | Mede Precisão/Recall da localização (IoU ≥ 0,5) contra os rótulos já revisados, sem envolver classificação vazia/com produto |
| `gerar_rotulos_yolo.py` | YOLO — pré-rotulagem | Usa o próprio pipeline clássico pra gerar rótulos YOLO iniciais (toda caixa nasce como `ampola_vazia`), só pra revisar/corrigir no LabelImg em vez de rotular do zero |
| `organizar_dataset_yolo.py` | YOLO — dataset | Separa as fotos rotuladas (e revisadas no LabelImg) em treino/validação e gera a estrutura de pastas + `data.yaml` que o `ultralytics` espera |

## Etapa 1 — Detecção clássica

### Passo 1: tampa (`detectar_tampas.py`)

Cada ampola tem uma tampa/anel preto numa das pontas, bem mais escura
que o vidro e o fundo branco da bandeja — isso dá um contraste alto e
fácil de isolar só com limiar de intensidade.

```bash
python detectar_tampas.py --image foto_exemplo.jpg --out resultado.png --debug
```

Isso gera:
- `resultado.png` — foto original com caixa verde + número em cada
  tampa detectada.
- `resultado_1_mascara_bruta.png` — tudo que passou no limiar de
  "escuro o suficiente". Se uma tampa não aparece branca aqui, o
  algoritmo nunca vai encontrá-la (os passos seguintes só filtram o que
  já está aqui, não recuperam o que faltou).
- `resultado_2_mascara_limpa.png` — depois de remover os blobs grandes
  (fundo escuro fora da bandeja).
- `resultado_3_watershed.png` — cada tampa detectada com uma cor
  diferente. Se duas tampas vizinhas aparecem com a mesma cor, o
  watershed não separou (viraram 1 detecção só). Se uma tampa só
  aparece com 2 cores, ele separou demais.

### Passo 2: ampola inteira (`detectar_ampolas.py`)

Tentativa de segmentar o contorno do vidro direto (Canny +
`connectedComponents`) foi abandonada: ampolas encostadas não geram
borda fechada entre si, e a textura do produto fragmenta o interior em
vários pedaços. O que funciona: usar a tampa (já detectada com
confiança) como âncora, agrupar por linha, medir a distância entre
tampas vizinhas na mesma linha (o "pitch" — equivale ao comprimento de
1 ampola) e expandir cada caixa de tampa por `pitch` pixels na direção
do corpo da ampola.

```bash
python detectar_ampolas.py --image foto_exemplo.jpg --out resultado_ampolas.png --pitch-direcao esquerda
```

### Rodar em lote

```bash
python processar_pasta.py --input caminho/para/suas_70_fotos/ --out resultados/
```

Imprime a contagem (tampa + ampola) de cada foto e salva a versão
anotada de todas em `resultados/`.

### Validação quantitativa

```bash
python validacao_classico.py
```

Roda o pipeline clássico contra os rótulos já revisados (formato YOLO)
e calcula Precisão/Recall de localização (pareamento por IoU ≥ 0,5).
Mede só se a caixa foi encontrada no lugar certo — não mede
classificação vazia/com produto (a abordagem clássica não distingue
isso; ver Etapa 2). 

## Guia de parâmetros — o que mexer quando

| Sintoma que você vê no debug | Parâmetro | O que fazer |
|---|---|---|
| Tampa não aparece branca na `mascara_bruta` | `--dark-threshold` | Aumentar (ex: 45 → 55) — mais permissivo pro que conta como "escuro" |
| Fundo/sombra aparecendo como branco na `mascara_bruta` | `--dark-threshold` | Diminuir (ex: 45 → 35) — mais exigente |
| Duas tampas vizinhas com a mesma cor no `watershed` (contadas como 1) | `--min-distance-picos` | Diminuir (ex: 12 → 8) |
| Uma tampa só aparece dividida em 2 cores no `watershed` | `--min-distance-picos` | Aumentar (ex: 12 → 16) |
| Caixa aparecendo em cima de grão de produto (não é tampa) | `--area-min` | Aumentar |
| Tampa real ficou sem caixa (foi descartada por área) | `--area-min` / `--area-max` | Olhar o tamanho real dela na `mascara_limpa` e ajustar a faixa |
| Produto/sombra virando "tampa" mesmo depois de mexer no `--dark-threshold` | `--intensidade-media-max` | Medir a escuridão média de uma tampa real e de um blob errado, e definir um valor entre os dois — ex: `--intensidade-media-max 27` |
| Caixa verde aparecendo em cima de um amontoado comprido/fino de produto | `--preenchimento-min` (ex: 0.5) | Tampa é um blob sólido; produto costuma preencher menos o retângulo |
| Tampa na borda da foto sumindo (2 pedacinhos pequenos demais cada um) | já é automático (`mesclar_caixas_proximas`) | Se ainda sumir, aumentar um pouco `--margem-juncao` (padrão 6) |
| Duas tampas vizinhas de verdade sendo contadas como 1 só | `--margem-juncao` | Diminuir (a junção só deveria valer pra pedaço pequeno demais pra ser tampa sozinho) |

**Sobre `--intensidade-media-max`:** cada foto pode ter iluminação
diferente, então esse valor não é universal — foi a principal fonte de
fragilidade do método clássico.
Mudar um parâmetro de cada vez e comparar numa foto "fácil" e numa
"difícil"; se os valores calibrados forem muito diferentes entre fotos,
é sinal de que um conjunto fixo de parâmetros não generaliza, essa foi
justamente a limitação que motivou a Etapa 2 (YOLO).

## Etapa 2 — Transição para YOLO

Fluxo adotado:

1. **Pré-rotulagem** (`gerar_rotulos_yolo.py`) — reaproveita o pipeline
   clássico pra gerar as caixas iniciais em formato YOLO; toda caixa
   nasce como `ampola_vazia`, já que o método clássico só localiza, não
   classifica.

   ```bash
   python gerar_rotulos_yolo.py --input dataset/fotos --intensidade-media-max 27
   ```

2. **Revisão manual no LabelImg** — corrige posição/tamanho e marca a
   classe certa (`ampola_vazia` / `ampola_com_produto`) em cada caixa.

3. **Organização do dataset** (`organizar_dataset_yolo.py`) — separa em
   treino/validação (80/20, semente fixa) e gera a estrutura de pastas
   + `data.yaml` que o `ultralytics` espera.

   ```bash
   python organizar_dataset_yolo.py --input dataset\ampolas --out dataset_yolo --val-frac 0.2
   ```

4. **Treino** — `yolo detect train data=dataset_yolo/data.yaml model=yolov8n.pt ...`


## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate     
pip install -r requirements.txt
```

`requirements.txt` cobre as duas etapas: OpenCV/NumPy/SciPy/scikit-image
para a parte clássica, `labelImg` para rotulagem e `ultralytics` para o
treino YOLO.

## Status atual / próximos passos

- Etapa 1 (clássica) concluída e documentada, incluindo a limitação de
  generalização do threshold que motivou a Etapa 2.
- Etapa 2 (YOLO): dataset rotulado e organizado; treino em andamento,
  mas resultados preliminares já estão dispostos na runs/detect/runs/detect/treino_completo