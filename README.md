# Inspeção visual de embalagens com YOLO

Trabalho de Conclusão de Curso (Capstone): sistema de visão computacional que
aponta embalagens vazias em bandejas de ampolas e blisters. O modelo YOLO
analisa a foto capturada pelo LabVIEW, marca as embalagens suspeitas para o
operador e a ordem de produção é integrada ao ERP, gerando um relatório por lote.

## Etapas do projeto

O repositório preserva as três etapas do Capstone, cada uma em sua pasta:

| Pasta | Etapa | Conteúdo |
|---|---|---|
| [`intermediario/`](intermediario/README.md) | Prova de conceito | Visão clássica (limiar + watershed) para ampolas, pré-rotulagem e primeiro teste com YOLO |
| `final/` | Desenvolvimento dos modelos | Datasets, treino e validação de um modelo YOLO por tipo de embalagem |
| `producao/` | Implantação | Inferência chamada pelo LabVIEW e integração com o ERP |

```text
capstone/
├── intermediario/          # scripts clássicos, dataset e testes da prova de conceito
├── final/
│   ├── dataset/            # fotos e rótulos originais
│   ├── dataset_yolo/       # split treino/val gerado (fora do Git)
│   ├── modelos/<tipo>/     # resultados de treino e validação de cada tipo
│   └── scripts_comuns/     # organizar_dataset, treinar, validar, inspecionar
├── producao/
│   ├── main.py             # inspeção de uma foto (chamado pelo LabVIEW)
│   ├── erp_doc.py          # consulta ao ERP e relatório do lote
│   ├── inspecionar.py      # inferência e marcação das embalagens vazias
│   └── modelos/<tipo>/     # best.pt em uso
├── .env.example            # modelo de configuração
└── requirements.txt        # ambiente completo (inclui as três etapas)
```

Tipos de embalagem: `ampola`, `blister_992`, `blister_bloco`, `blister_org` e
`blister_org2`.

## Instalação

Requer Python 3.12. No Windows (PowerShell), a partir da raiz do repositório:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Para instalar uma única etapa, use o `requirements.txt` da pasta dela. No
computador do LabVIEW basta o de `producao/`.

## Configuração

A chave da API do ERP fica em um arquivo `.env`, fora do Git:

```powershell
Copy-Item .env.example .env
```

Preencha `ERP_API_KEY` com a chave enviada e confira a `ERP_API_URL`. Se só a pasta `producao/`
for instalada, coloque o `.env` dentro dela.

| Variável | Obrigatória | Descrição |
|---|---|---|
| `ERP_API_KEY` | Sim | Chave de acesso à API do ERP |
| `ERP_API_URL` | Sim | URL da consulta de ordem; deve conter `{numero_da_ordem}` |
| `ERP_API_TIMEOUT_S` | Não | Tempo máximo de espera da API, em segundos (padrão: 10) |

## Etapa final: treinar e validar

Os comandos rodam a partir da raiz. Primeiro gere o split de treino/validação
e depois treine e valide o tipo desejado:

```powershell
python final/scripts_comuns/organizar_dataset.py --input final/dataset/DATASET_ampola --out final/dataset_yolo/ampola --val-frac 0.2
python final/scripts_comuns/treinar.py --tipo ampola --epochs 100 --nome treino_completo_v2
python final/scripts_comuns/validar.py --tipo ampola --pesos treino_completo_v2/weights/best.pt
```

Os resultados ficam em `final/modelos/<tipo>/`. Para colocar um modelo em uso,
copie o `best.pt` escolhido para `producao/modelos/<tipo>/best.pt`.

## Produção: interface com o LabVIEW

Os scripts são chamados pelo `System Exec.vi` e sempre devolvem uma linha no
stdout, com `;` depois de cada campo. Detalhes de erro vão para o stderr.

| Comando | Saída com sucesso | Saída com falha |
|---|---|---|
| `python main.py "<...>\Ampola\foto.png"` | `<total>;<vazias>;OK;` (ou `ERRADO` se houver vazias) | `0;0;ERRADO;` |
| `python erp_doc.py iniciar <ordem>` | `<lote>;<quantidade>;OK;` | `;0;ERRADO;` |
| `python erp_doc.py registrar <ordem> <vazias>` | `<total_vazias>;OK;` | `0;ERRADO;` |
| `python erp_doc.py finalizar <ordem>` | `<caminho_do_txt>;OK;` | `;ERRADO;` |

O tipo de embalagem é deduzido do nome da pasta da foto, que deve ser igual ao
nome em `producao/modelos/<tipo>/` (sem diferenciar maiúsculas). A imagem
anotada é salva na pasta irmã `<Pasta>_inspecionada`, com o mesmo nome de
arquivo. Os registros das ordens ficam em `producao/registros/` e os relatórios
em `producao/relatorios/`.
