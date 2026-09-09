# Passo 1 — Detecção de ampolas com visão computacional clássica

## Como iterar (o "loop" de calibração)

```
1. Rode numa foto com --debug
2. Abra as 3 imagens de debug geradas + a foto original, lado a lado
3. Decida o que ajustar (guia de parâmetros abaixo)
4. Rode de novo com o novo valor
5. Quando estiver bom nessa foto, rode em várias (processar_pasta.py)
6. Veja em quais quebra, volte ao passo 3
```

```bash
python python detectar_ampolas.py --image foto_teste.jpg --out resultado_ampolas.png --intensidade-media-max 27
```

Isso gera:
- `resultado.png` — foto original com caixa verde + número em cada
  ampola detectada. **Olhe aqui primeiro**: a contagem bate com a foto?
- `resultado_1_mascara_bruta.png` — tudo que passou no limiar de
  "escuro o suficiente". Se uma tampa não aparece BRANCA aqui, o
  algoritmo nunca vai encontrá-la (os passos seguintes só filtram o que
  já está aqui, não recuperam o que faltou).
- `resultado_2_mascara_limpa.png` — depois de remover os blobs grandes
  (fundo escuro fora da bandeja). Confira se sobrou fundo indevido, ou
  se limpou tampa demais.
- `resultado_3_watershed.png` — cada tampa detectada com uma cor
  diferente. Se duas tampas vizinhas aparecem com a MESMA cor, o
  watershed não separou (viram 1 detecção só). Se uma tampa só aparece
  com 2 cores, ele separou demais (viram 2 detecções falsas).

## Guia de parâmetros — o que mexer quando

| Sintoma que você vê no debug | Parâmetro | O que fazer |
|---|---|---|
| Tampa não aparece branca na `mascara_bruta` | `--dark-threshold` | Aumentar (ex: 45 → 55) — mais permissivo pro que conta como "escuro" |
| Fundo/sombra aparecendo como branco na `mascara_bruta` | `--dark-threshold` | Diminuir (ex: 45 → 35) — mais exigente |
| Duas tampas vizinhas com a mesma cor no `watershed` (contadas como 1) | `--min-distance-picos` | Diminuir (ex: 12 → 8) |
| Uma tampa só aparece dividida em 2 cores no `watershed` | `--min-distance-picos` | Aumentar (ex: 12 → 16) |
| Caixa verde aparecendo em cima de grão de produto (não é tampa) | `--area-min` | Aumentar |
| Tampa real ficou sem caixa verde (foi descartada por área) | `--area-min` / `--area-max` | Olhar o tamanho real dela na `mascara_limpa` e ajustar a faixa |
| Produto/sombra virando "tampa" mesmo depois de mexer no `--dark-threshold` | `--intensidade-media-max` | Meça a escuridão média de uma tampa real e de um blob errado (script de calibração, ver abaixo) e defina um valor entre os dois — ex: `--intensidade-media-max 27` |
| Caixa verde aparecendo em cima de um amontoado comprido/fino de produto | `--preenchimento-min` (ex: 0.5) | Tampa é um blob sólido; produto costuma preencher menos o retângulo |
| Tampa na borda da foto sumindo (2 pedacinhos pequenos demais cada um) | já é automático (`mesclar_caixas_proximas`) | Se ainda sumir, aumente um pouco `--margem-juncao` (padrão 6) |
| Duas tampas vizinhas de verdade sendo contadas como 1 só | `--margem-juncao` | Diminuir (a junção só deveria valer pra pedaço pequeno demais pra ser tampa sozinho) |

### Como calibrar `--intensidade-media-max` pra uma foto nova

Cada foto pode ter uma iluminação um pouco diferente, então esse número não é universal. Jeito rápido de achar o valor certo pra uma foto específica:

```bash
python detectar_tampas.py --image sua_foto.jpg --out /tmp/teste.png --debug
```

Abra `/tmp/teste_3_watershed.png` ao lado da foto original. Ache 2-3 tampas de verdade e 2-3 blobs errados (produto virando "tampa"). Se quiser o número exato em vez de chutar, peça pra eu escrever um scriptzinho de medição (é o que eu fiz pra achar o 27 dessa sua foto) — é rápido.

Mude **um parâmetro de cada vez** e rode de novo — assim você sabe qual
mudança causou qual efeito. Anote os valores que funcionaram bem numa
foto "fácil" (boa iluminação, sem oclusão) e numa foto "difícil" (mão na
frente, ampola torta) — se forem muito diferentes, é sinal de que um
único conjunto de parâmetros fixos não vai servir pro dataset inteiro
(limitação conhecida de visão clássica pura — vale documentar no TCC).

## Testar em várias fotos de uma vez

```bash
python processar_pasta.py --input dataset/ --out resultados/
```

Imprime a contagem de cada foto e salva a versão anotada de todas em
`resultados/`. Foque nas fotos com contagem muito diferente da média —
são os casos que vão te ensinar o que ajustar.

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Próximo passo (ainda não feito)

Depois que a detecção estiver estável na maioria das fotos: para cada
caixa detectada, medir uma estatística simples da região (variância ou
nível médio de cinza) pra decidir presença/ausência — o mesmo princípio
visual que você já usa a olho nu (produto = textura/escurecimento
irregular; vazio = vidro uniforme e claro). Fazemos isso quando você
sentir que a contagem está confiável.