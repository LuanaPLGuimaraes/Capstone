# Contexto técnico — detecção de ampolas com visão computacional clássica

Guia de estudo pra você levar pra reunião com o orientador. Estrutura: o
que foi decidido, por quê, os números exatos usados, o que já tentamos e
não funcionou, e perguntas prováveis com resposta pronta.

---

## 1. Enquadramento do problema

**Restrição do TCC**: só visão computacional CLÁSSICA — sem deep learning,
sem dataset de treino, sem rede neural. Isso significa que **nada aqui
"aprende" com as fotos**. Todo o comportamento vem de regras de
processamento de imagem (limiarização, morfologia, watershed) cujos
parâmetros nós calibramos olhando os resultados — não existe fase de
treinamento nem função de perda.

**Entrada**: fotos 1920×1080 de uma bandeja plástica branca, vista de
cima, com ampolas de vidro deitadas na horizontal, encostadas umas nas
outras, formando uma grade (~6 colunas × ~12 linhas). Câmera e ângulo
fixos; a disposição das ampolas varia de foto pra foto (de propósito —
simula variações do chão de fábrica).

**Por que é difícil**: o vidro é transparente sobre fundo branco →
contraste quase zero na borda do corpo da ampola. A única feição de alto
contraste e confiável é o anel/tampa preto numa das pontas de cada
ampola. Isso definiu a estratégia inteira: **usar a tampa como âncora**,
não tentar segmentar o vidro diretamente.

**Objetivo desta etapa**: localizar e contar as ampolas (delimitar a
caixa da tampa e depois da ampola inteira). Presença/ausência de produto
(a classificação em si) é a próxima etapa, ainda não implementada.

---

## 2. Pipeline — visão geral

```
foto original (BGR)
   → escala de cinza
   → limiarização (threshold)          → "máscara bruta"
   → abertura morfológica (ruído)
   → remoção de blobs grandes (fundo)   → "máscara limpa"
   → watershed (separa tampas encostadas)
   → 1 caixa por região do watershed
   → JUNTAR fragmentos pequenos grudados (antes de filtrar por área)
   → filtrar por área / formato / preenchimento / intensidade média
   → [caixas da TAMPA prontas]
   → agrupar tampas por linha
   → medir espaçamento da grade (pitch) por linha
   → expandir cada tampa pelo pitch, na direção do corpo
   → [caixas da AMPOLA INTEIRA prontas]
```

Etapa 1 (até "caixas da tampa prontas") está em `detectar_tampas.py`.
Etapa 2 (o resto) está em `detectar_ampolas.py`, que importa e reusa a
etapa 1.

---

## 3. Etapa 1 — Detecção da tampa

### 3.1 O que são "aquelas fotos em preto e branco"

São as 3 imagens de debug (`--debug`), uma por estágio intermediário do
pipeline — não são a saída final, são pra diagnosticar onde o algoritmo
está errando:

| Imagem | O que mostra | Pra que serve olhar |
|---|---|---|
| `mascara_bruta` | Resultado cru do threshold: todo pixel abaixo do limiar vira branco, o resto preto | Confere se a tampa aparece branca (contraste suficiente) |
| `mascara_limpa` | A mesma coisa, depois de remover blobs enormes (fundo fora da bandeja) e ruído pequeno | Confere se sobrou fundo indevido ou se limpou tampa demais |
| `watershed` | Cada blob final pintado com uma cor aleatória diferente | Confere se tampas vizinhas foram separadas (cores diferentes) ou se uma tampa foi dividida em 2 (2 cores na mesma tampa) |

São todas imagens **binárias/rotuladas**, não fotos normais — por isso
o aspecto preto e branco (ou "manchado colorido" no caso do watershed).

### 3.2 Limiarização (threshold)

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mascara_bruta = cv2.threshold(gray, dark_threshold=45, 255, cv2.THRESH_BINARY_INV)
```

Binarização simples e global: todo pixel com intensidade de cinza **abaixo**
de 45 (numa escala 0–255) vira branco (candidato a tampa); o resto,
preto. `THRESH_BINARY_INV` porque queremos o **escuro** como
"positivo". Não é Otsu nem adaptativo — é um limiar fixo, calibrado
manualmente olhando o histograma de intensidade da tampa vs. fundo.
**Isso é a maior fragilidade do método** (ver seção 6).

### 3.3 Limpeza morfológica + remoção de blobs grandes

```python
mask = cv2.morphologyEx(mascara_bruta, cv2.MORPH_OPEN, kernel_3x3, iterations=1)
n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
# descarta qualquer componente conexo com área >= area_max_blob_fundo (5000 px)
```

Abertura morfológica = erosão seguida de dilatação. Explicando cada uma
(pensando na máscara como uma grade de pixels branco/preto, e a
operação olhando sempre o pixel + os 8 vizinhos, janela 3×3):

- **Erosão**: um pixel só continua branco se TODOS os vizinhos também
  forem brancos. Corrói a borda de toda região branca pra dentro (tipo
  maré corroendo a praia). Efeito: uma manchinha de 1-2 pixels (ruído)
  não tem "miolo" suficiente pra sobreviver e desaparece por completo;
  um blob grande (tampa de verdade) só encolhe um pouco nas bordas.
- **Dilatação**: o oposto — um pixel vira branco se PELO MENOS UM
  vizinho for branco. Expande toda região branca pra fora. Efeito:
  devolve o tamanho original de quem sobreviveu à erosão.

Por que as duas juntas (nessa ordem): erosão sozinha destruiria ruído
mas também encolheria as tampas de verdade; aplicando dilatação depois,
o ruído (que já foi zerado) não tem como voltar, e a tampa (que só
encolheu, não sumiu) recupera o tamanho. Resultado líquido: ruído
removido, tampas preservadas.

Depois disso, `connectedComponentsWithStats` rotula cada região conexa
que sobrou e descartamos qualquer uma maior que 5000 px — isso elimina
cantos escuros de fundo fora da bandeja (que sem esse corte viravam um
blob de dezenas de milhares de pixels).

### 3.4 Watershed — separar tampas encostadas

Problema: tampas vizinhas costumam se tocar na foto e o threshold sozinho
funde as duas num blob só.

```python
dist = ndi.distance_transform_edt(mascara_limpa)          # transformada de distância
coords = peak_local_max(dist, min_distance=12, labels=mascara_limpa)  # picos locais = 1 por tampa
markers, _ = ndi.label(peak_mask)
ws = watershed(-dist, markers, mask=mascara_limpa)
```

Ideia clássica de watershed por transformada de distância: cada pixel
branco recebe o valor da distância até o pixel preto mais próximo — o
"miolo" de cada tampa vira um pico local nessa superfície. Cada pico
detectado (com `min_distance_picos=12` px de separação mínima entre
picos) vira uma "semente"; o watershed inunda a partir dessas sementes e
para exatamente na fronteira entre duas tampas que se tocam. Resultado:
2 tampas encostadas saem como 2 regiões, não 1.

`min_distance_picos` é o parâmetro mais sensível dessa etapa: baixo
demais separa demais (1 tampa vira 2); alto demais não separa (2 tampas
viram 1).

### 3.5 Filtros de validação (o que faz um blob "ser" uma tampa)

Depois do watershed, cada região vira uma caixa candidata. Nem toda
região é uma tampa de verdade — grãos de produto também são escuros e
cruzam o mesmo limiar. Quatro filtros, todos calibráveis, aplicados
nessa ordem:

| Filtro | Parâmetro | Lógica |
|---|---|---|
| Área | `area_min=500`, `area_max=2300` (px²) | Tampa tem um tamanho esperado; blob menor é ruído/grão, maior é fusão de 2+ objetos |
| Formato | `altura_min/max`, `largura_min/max` (px) | Um amontoado de grãos pode ter a MESMA área de uma tampa mas formato bem diferente (mais comprido/fino) — desligado por padrão, calibrável por foto |
| Preenchimento (extent) | `preenchimento_min` = área / (largura×altura) | Tampa é um blob sólido (preenche quase todo o retângulo); textura de produto forma um blob esburacado — desligado por padrão |
| Intensidade média | `intensidade_media_max` | Em vez de só "cruzou o limiar", confere a média de cinza de TODOS os pixels do blob. Ver números reais abaixo |

**Números medidos numa foto real do dataset** (script de calibração,
medindo blob a blob): tampas verdadeiras ficaram com intensidade média
entre **~8 e ~26**; blobs de produto confundidos com tampa ficaram entre
**~28 e ~40**. Existe uma "fresta" clara ali no meio — por isso
calibramos `intensidade_media_max=27` pra essa foto. Esse valor
**não é universal**: cada condição de iluminação pode deslocar essa
fresta, por isso é recalibrado foto a foto (ou por lote de fotos com
iluminação parecida).

### 3.6 Correção de fragmentação (tampa cortada em 2 pedaços)

Problema real observado: o watershed às vezes corta 1 tampa real (geralmente
na coluna da borda da imagem) em 2 pedaços pequenos, cada um menor que
`area_min` sozinho — e cada um seria descartado individualmente,
perdendo a tampa inteira.

Solução: juntar pedaços **antes** de aplicar o filtro de área, com uma
regra de 2 partes pra não desfazer o trabalho do watershed:

1. **Sobreposição forte** (IoU ≥ 0.25): sempre junta. Nunca é 2 tampas
   reais diferentes (elas nunca ocupam o mesmo espaço).
2. **Só grudadas, sem sobrepor** (distância ≤ `margem_juncao=6` px):
   só junta se **pelo menos um** dos dois pedaços for menor que
   `area_min` sozinho. Essa condição é essencial: tampas vizinhas de
   verdade também ficam grudadas (é por isso que o watershed existe!)
   — se juntássemos qualquer par grudado, desfaríamos a separação e
   contaríamos 2 tampas reais como 1.

(IoU = interseção sobre união, métrica padrão de sobreposição entre
bounding boxes, 0 a 1.)

---

## 4. Etapa 2 — Delimitação da ampola inteira (não só a tampa)

### 4.1 Tentativa que NÃO funcionou (documentar isso é importante)

Primeira ideia: achar o contorno inteiro da ampola direto, via Canny
(detecção de bordas) → dilatar → `connectedComponents` → watershed de
novo pra separar cada ampola.

**Por que falhou**: como as ampolas ficam encostadas, o "limite" entre
duas vizinhas não fecha como um contorno sólido — Canny não gera uma
borda contínua ali. Resultado: a segmentação juntava ampola com ampola
ou cortava uma ampola em vários pedaços por causa da textura do produto
dentro dela (cada grão virava uma micro-borda). Caixas finais saíam
tortas, sobrepostas e com contagem errada.

**Conclusão registrada**: segmentar objetos transparentes e encostados
só por contorno de borda é inviável sem uma pista de contraste adicional
— é uma limitação conhecida, não um bug de implementação.

### 4.2 Método que funciona: âncora + espaçamento de grade

Como a tampa já é detectada com confiança (etapa 1), usamos ela como
**âncora geométrica** e a regularidade da grade pra saber o tamanho da
ampola, em vez de tentar "ver" o vidro:

1. **Agrupar tampas por linha** (`agrupar_por_linha`): agrupa caixas de
   tampa com centro em Y parecido (tolerância de 25 px) — cada grupo é
   uma fileira física de ampolas.
2. **Medir o pitch** (`medir_pitch`): dentro de cada linha, calcula a
   distância entre os centros X de tampas vizinhas. Como as ampolas
   ficam encostadas ponta a ponta, essa distância ≈ comprimento de 1
   ampola inteira. Usa a **mediana** dessas distâncias (robusta a
   outliers), descartando valores fora de uma faixa esperada
   (`pitch_min=100`, `pitch_max=350` px — muito perto costuma ser tampa
   partida ainda não corrigida; muito longe costuma ser uma tampa
   faltando na grade, não o espaçamento real). Valor medido nas fotos
   testadas: **pitch ≈ 190–200 px**.
3. **Expandir** (`expandir_para_ampola`): cada caixa de tampa vira a
   caixa da ampola inteira, estendendo `pitch` pixels na direção do
   corpo (checada visualmente por foto — pode ser "esquerda" ou
   "direita").

Essa abordagem só é válida pro **Cenário 1** (ampolas deitadas, mesma
orientação, grade regular mesmo que incompleta). Ampolas espalhadas ou
giradas (Cenário 2) ficam pra depois — provavelmente vai exigir
`cv2.minAreaRect` ou outra abordagem sensível a rotação.

---

## 5. Calibração e fluxo de trabalho

Como nada aqui aprende com dados, calibrar = olhar o resultado visual +
as 3 imagens de debug e ajustar 1 parâmetro por vez. Fluxo usado:

```
1. Rodar numa foto com --debug
2. Comparar as 3 imagens de debug + resultado final com a foto original
3. Decidir o que ajustar (tabela de sintoma → parâmetro no README)
4. Rodar de novo
5. Quando bom numa foto, rodar no lote inteiro (processar_pasta*.py)
6. Ver em quais quebra, voltar ao passo 3
```

As 70 fotos do dataset **não treinam nada** — servem pra testar se um
conjunto de parâmetros generaliza (teste de robustez), não pra ajustar
pesos de um modelo.

---

## 6. Limitação central, em uma frase

O método inteiro depende de threshold de intensidade calibrado por foto
— é a raiz de quase tudo que ainda quebra (iluminação diferente desloca
o corte ideal) e o principal argumento pra discussão da seção 7 (trazer
YOLO já nessa etapa ou não). Detalhes técnicos de cada limitação estão
nas seções acima; a lista de pontos de decisão está na seção 7.

---

## 7. Pontos pra discutir com o orientador

Isso aqui não é lista de "prova" — as decisões técnicas foram tomadas
junto, passo a passo. É mais um roteiro do que vale a pena colocar na
mesa pra ele opinar, já que ele é especialista e pode enxergar caminhos
que a gente não viu.

**O processo até aqui, resumido**: começamos tentando achar a borda do
vidro direto (Canny) e não deu certo (seção 4.1) — vidro transparente
sobre fundo branco não gera contorno fechado onde as ampolas se tocam,
e a textura do produto fragmenta tudo. Pivotamos pra usar a tampa (alto
contraste) como âncora e a regularidade da grade (espaçamento entre
tampas) pra inferir o tamanho da ampola inteira. Funciona bem pra grade
alinhada (Cenário 1); ainda não cobre ampolas espalhadas/giradas
(Cenário 2).

**A ideia principal pra levar**: será que vale a pena colocar um YOLO
(ou outro modelo treinado) já nessa etapa de localização/contagem, em
vez de deixar pra usar deep learning só depois, na classificação de
presença/ausência? Pontos concretos pra essa conversa:

- **A favor de manter clássico aqui**: já funciona razoavelmente bem no
  Cenário 1, sem precisar de dataset rotulado nem GPU — e documentar
  essa baseline clássica (o que funciona, o que não funciona e por quê)
  tem valor de TCC por si só, como comparação.
- **A favor de trazer YOLO já pra essa etapa**: resolveria de uma vez a
  maior fragilidade que encontramos (dependência de threshold calibrado
  por foto — muda a iluminação, precisa recalibrar) e também cobriria o
  Cenário 2 (ampolas giradas/espalhadas) sem precisar inventar uma
  segunda lógica geométrica pra isso. O "custo" é precisar rotular
  manualmente uma boa quantidade das 70 fotos como dataset de treino —
  o que já estava nos nossos planos como preparação pra etapa seguinte
  de qualquer forma.
- **Meio-termo possível**: usar o pipeline clássico pra gerar rótulos
  iniciais automaticamente (bootstrap) nas fotos do Cenário 1, revisar
  à mão, e treinar um YOLO leve com isso — reduz o trabalho manual de
  rotular do zero.

**Outras limitações que valem menção**, pra ele já ter o quadro
completo:
- Threshold fixo (não Otsu/adaptativo) — funciona mas exige recalibrar
  por condição de luz.
- Direção do corpo da ampola é fixada manualmente por foto
  (`--pitch-direcao`), não inferida automaticamente ainda.
- Meta do projeto é 99,99% de acerto (benchmark de mercado ~80%); essa
  etapa é só localização/contagem — acurácia real de presença/ausência
  ainda não foi medida (falta gabarito rotulado manualmente).
