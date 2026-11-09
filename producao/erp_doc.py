"""
PRODUÇÃO — INTEGRAÇÃO ERP (TOTVS Datasul) + RELATÓRIO TXT DO LOTE.

Fluxo de uma ordem (o LabVIEW chama via System Exec.vi, igual ao main.py):

  1) INICIAR — quando o operador faz login e informa a ordem:
        python erp_doc.py iniciar <numero_ordem>
     - consulta a API do ERP com o número da ordem;
     - recebe a quantidade aprovada da ordem (o "lote" é o próprio
       número da ordem — a API não devolve um número de lote separado);
     - grava o horário de login (o horário DESTA chamada);
     - abre um registro da ordem em registros/ordem_<ordem>.json.
     Saída (uma linha, mesmo padrão do main.py, ";" depois de cada campo):
        <numero_lote>;<quantidade_lote>;OK;
        ;0;ERRADO;        -> falhou (API fora, ordem não achada, etc.)

  2) REGISTRAR — depois de CADA bandeja processada pelo main.py:
        python erp_doc.py registrar <numero_ordem> <total_vazias>
     - <total_vazias> é o 2º campo que o main.py devolveu;
     - se for > 0, grava o horário em que a ausência foi detectada
       naquela bandeja e quantas vazias ela tinha;
     - se for 0, não grava nada (só confirma OK).
     Saída:
        <total_vazias_acumulado_na_ordem>;OK;
        0;ERRADO;

  3) FINALIZAR — quando a ordem termina:
        python erp_doc.py finalizar <numero_ordem>
     - gera o txt em relatorios/ com: número do lote, ordem, código e
       descrição do item, quantidade inspecionada (quantidade do lote -
       total de vazias), horário de login e o horário de cada bandeja em
       que houve ausência;
     - fecha o registro da ordem (renomeia para .finalizado.json).
     Saída:
        <caminho_do_txt>;OK;
        ;ERRADO;

Assim como no main.py: qualquer erro vira ERRADO no stdout e o detalhe
vai para o stderr (só para debug manual).

API (informada pela equipe de TI):
    GET <ERP_API_URL, com o número da ordem em {numero_da_ordem}>
    Cabeçalho: X-Api-Key: <chave>
    Resposta 200 (exemplo real, ordem 979993201):
        {"nr_ord_prod": 979993201, "codigo_item": "520.10",
         "descricao_item": "FIXADOR TIPO COLLES P/ PUNHO",
         "quantidade_aprovada": 49.0000}
    Erros observados: 404 (ordem não encontrada, corpo com "mensagem"),
    400 (número da ordem inválido), 401 (chave ausente/errada).

CHAVE DA API — NÃO fica neste arquivo (o repositório é público). Ela é
lida, nesta ordem:
    1) variável de ambiente ERP_API_KEY, que pode vir de um arquivo .env
       (ignorado pelo Git) em producao/ ou na raiz do projeto. Modelo em
       .env.example; precisa ser criado em cada computador; ou
    2) arquivo legado erp_config.json ao lado deste script:
           {"api_chave": "<chave enviada pela TI>"}
ERP_API_URL (obrigatória) e ERP_API_TIMEOUT_S (opcional) também vêm do .env.
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # sem o pacote, ainda valem as variáveis de ambiente do sistema
    load_dotenv = None

# ============================================================
# CONFIGURAÇÃO DA API (URL e chave vêm do .env)
# ============================================================
API_CABECALHO_CHAVE = "X-Api-Key"
API_TIMEOUT_S = 10  # segundos esperando a resposta antes de desistir

# Nomes dos campos no JSON de resposta
CAMPO_ORDEM = "nr_ord_prod"            # também usado como número do lote
CAMPO_CODIGO_ITEM = "codigo_item"
CAMPO_DESCRICAO_ITEM = "descricao_item"
CAMPO_QUANTIDADE = "quantidade_aprovada"
# ============================================================

AQUI = Path(__file__).resolve().parent
ARQUIVO_CONFIG = AQUI / "erp_config.json"  # guarda a chave; fora do Git
PASTA_REGISTROS = AQUI / "registros"       # estado das ordens em andamento
PASTA_RELATORIOS = AQUI / "relatorios"     # txts gerados
FORMATO_HORA = "%d/%m/%Y %H:%M:%S"

if load_dotenv:
    # producao/.env tem prioridade sobre o da raiz; variáveis já definidas
    # no sistema têm prioridade sobre ambos. utf-8-sig: aceita o BOM do
    # Bloco de Notas.
    load_dotenv(AQUI / ".env", encoding="utf-8-sig")
    load_dotenv(AQUI.parent / ".env", encoding="utf-8-sig")


# ------------------------------------------------------------
# Integração ERP
# ------------------------------------------------------------
def _validar_numero_ordem(numero_ordem: str) -> str:
    """A API só aceita número (texto devolve HTTP 400). O número também
    vira parte do nome dos arquivos de registro/relatório, então barrar
    qualquer outra coisa aqui evita caminhos estranhos (ex.: '..\\x')."""
    numero_ordem = str(numero_ordem).strip()
    if not numero_ordem.isdigit() or not numero_ordem.isascii():
        raise ValueError(f"Número de ordem inválido: '{numero_ordem}' (só dígitos)")
    return numero_ordem


def _ler_chave_api() -> str:
    chave = os.environ.get("ERP_API_KEY", "").strip()
    if chave:
        return chave
    if not ARQUIVO_CONFIG.exists():
        raise RuntimeError("Chave da API não configurada: defina ERP_API_KEY no .env (modelo em .env.example)")
    # utf-8-sig: aceita o arquivo mesmo se o Bloco de Notas salvar com BOM
    config = json.loads(ARQUIVO_CONFIG.read_text(encoding="utf-8-sig"))
    chave = str(config.get("api_chave", "")).strip()
    if not chave or "COLOQUE_AQUI" in chave:
        raise RuntimeError(f"Campo 'api_chave' vazio em {ARQUIVO_CONFIG}")
    return chave


def _mensagem_erro_api(erro: urllib.error.HTTPError) -> str:
    """Tenta extrair a mensagem que a API manda no corpo do erro."""
    try:
        corpo = json.loads(erro.read().decode("utf-8"))
    except Exception:
        return ""
    if isinstance(corpo, dict):
        return str(corpo.get("mensagem") or corpo.get("title") or "")
    return ""


def _converter_quantidade(valor) -> int:
    """A API manda a quantidade como decimal (ex.: 49.0000). Converte para
    inteiro, mas recusa valores com fração ou negativos em vez de
    arredondar em silêncio."""
    try:
        quantidade = Decimal(str(valor))
    except (InvalidOperation, ValueError) as erro:
        raise RuntimeError(f"Quantidade inválida na resposta da API: {valor!r}") from erro
    if not quantidade.is_finite() or quantidade != quantidade.to_integral_value() or quantidade < 0:
        raise RuntimeError(f"Quantidade da API não é um inteiro >= 0: {valor!r}")
    return int(quantidade)


def consultar_ordem(numero_ordem: str) -> dict:
    """Consulta a API do ERP e devolve
    {'lote', 'ordem', 'codigo_item', 'descricao_item', 'quantidade'}."""
    numero_ordem = _validar_numero_ordem(numero_ordem)
    chave = _ler_chave_api()
    timeout_s = float(os.environ.get("ERP_API_TIMEOUT_S") or API_TIMEOUT_S)
    url_ordem = os.environ.get("ERP_API_URL", "").strip()
    if not url_ordem:
        raise RuntimeError("URL da API não configurada: defina ERP_API_URL no .env (modelo em .env.example)")

    url = url_ordem.format(numero_da_ordem=urllib.parse.quote(numero_ordem))
    requisicao = urllib.request.Request(
        url,
        method="GET",
        headers={API_CABECALHO_CHAVE: chave, "Accept": "application/json"},
    )

    try:
        with urllib.request.urlopen(requisicao, timeout=timeout_s) as resposta:
            # parse_float=Decimal: mantém "49.0000" exato para a checagem de inteiro
            dados = json.loads(resposta.read().decode("utf-8"), parse_float=Decimal)
    except urllib.error.HTTPError as erro:
        detalhe = _mensagem_erro_api(erro)
        if erro.code == 404:
            motivo = "ordem não encontrada no ERP"
        elif erro.code == 401:
            motivo = "chave da API recusada (confira ERP_API_KEY no .env)"
        elif erro.code == 400:
            motivo = "número de ordem recusado pela API"
        else:
            motivo = "erro na API"
        raise RuntimeError(f"HTTP {erro.code} para a ordem {numero_ordem}: {motivo}"
                           + (f" ({detalhe})" if detalhe else "")) from erro
    except urllib.error.URLError as erro:
        raise RuntimeError(f"Não consegui conectar na API: {erro.reason}") from erro
    except TimeoutError as erro:
        raise RuntimeError(f"API não respondeu em {timeout_s}s") from erro
    except json.JSONDecodeError as erro:
        raise RuntimeError("Resposta da API não é um JSON válido") from erro

    if not isinstance(dados, dict):
        raise RuntimeError(f"Resposta da API em formato inesperado: {dados!r}")
    faltando = [c for c in (CAMPO_ORDEM, CAMPO_QUANTIDADE) if dados.get(c) is None]
    if faltando:
        raise RuntimeError(f"Resposta da API sem os campos {faltando}: {dados}")

    ordem_api = str(dados[CAMPO_ORDEM]).strip()
    if ordem_api != numero_ordem.lstrip("0"):
        raise RuntimeError(f"API devolveu a ordem {ordem_api}, mas foi pedida a {numero_ordem}")

    return {
        "lote": ordem_api,  # a API não tem número de lote: o lote é a ordem
        "ordem": ordem_api,
        "codigo_item": str(dados.get(CAMPO_CODIGO_ITEM) or "").strip(),
        "descricao_item": str(dados.get(CAMPO_DESCRICAO_ITEM) or "").strip(),
        "quantidade": _converter_quantidade(dados[CAMPO_QUANTIDADE]),
    }


# ------------------------------------------------------------
# Registro da ordem em andamento (persiste entre as chamadas)
# ------------------------------------------------------------
def _caminho_registro(numero_ordem: str) -> Path:
    return PASTA_REGISTROS / f"ordem_{numero_ordem}.json"


def _ler_registro(numero_ordem: str) -> dict:
    caminho = _caminho_registro(numero_ordem)
    if not caminho.exists():
        raise RuntimeError(f"Ordem {numero_ordem} não foi iniciada (falta chamar 'iniciar')")
    return json.loads(caminho.read_text(encoding="utf-8"))


def _salvar_registro(registro: dict) -> None:
    PASTA_REGISTROS.mkdir(parents=True, exist_ok=True)
    caminho = _caminho_registro(registro["ordem"])
    temporario = caminho.with_suffix(".tmp")
    # grava num temporário e renomeia: se cair a energia no meio, o
    # registro antigo não fica corrompido
    temporario.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    temporario.replace(caminho)


def iniciar_ordem(numero_ordem: str) -> dict:
    """Consulta o ERP, grava o horário de login e abre o registro da ordem."""
    numero_ordem = _validar_numero_ordem(numero_ordem)
    if _caminho_registro(numero_ordem).exists():
        # não sobrescreve: perderia as ausências já registradas
        raise RuntimeError(f"Ordem {numero_ordem} já está em andamento (finalize antes de reiniciar)")

    horario_login = datetime.now().strftime(FORMATO_HORA)
    dados_erp = consultar_ordem(numero_ordem)

    registro = {
        # chave do registro = o que o LabVIEW digitou (igual nas próximas
        # chamadas 'registrar'/'finalizar', mesmo com zeros à esquerda)
        "ordem": numero_ordem,
        "lote": dados_erp["lote"],
        "codigo_item": dados_erp["codigo_item"],
        "descricao_item": dados_erp["descricao_item"],
        "quantidade_lote": dados_erp["quantidade"],
        "horario_login": horario_login,
        "ausencias": [],  # [{"horario": ..., "vazias": n}, ...] — uma por bandeja com ausência
    }
    _salvar_registro(registro)
    return registro


def registrar_bandeja(numero_ordem: str, total_vazias: int, horario: datetime = None) -> dict:
    """Registra o resultado de uma bandeja. Só grava horário se houve ausência.
    `horario` permite passar a hora exata do processamento da imagem (quando
    for chamado de dentro do main.py); se omitido, usa a hora atual."""
    numero_ordem = _validar_numero_ordem(numero_ordem)
    if total_vazias < 0:
        raise ValueError(f"total_vazias inválido: {total_vazias}")

    registro = _ler_registro(numero_ordem)
    if total_vazias > 0:
        horario = horario or datetime.now()
        registro["ausencias"].append({"horario": horario.strftime(FORMATO_HORA), "vazias": total_vazias})
        _salvar_registro(registro)
    return registro


def _total_vazias(registro: dict) -> int:
    return sum(a["vazias"] for a in registro["ausencias"])


# ------------------------------------------------------------
# Documento TXT
# ------------------------------------------------------------
def gerar_txt(registro: dict, carimbo: str) -> Path:
    """Gera o relatório txt do lote e devolve o caminho do arquivo."""
    total_vazias = _total_vazias(registro)
    quantidade_inspecionada = registro["quantidade_lote"] - total_vazias
    if quantidade_inspecionada < 0:
        print(f"AVISO: mais vazias ({total_vazias}) do que a quantidade do lote "
              f"({registro['quantidade_lote']})", file=sys.stderr)

    linhas = [
        f"Número do lote: {registro['lote']}",
        f"Ordem de produção: {registro['ordem']}",
        f"Código do item: {registro.get('codigo_item', '')}",
        f"Descrição do item: {registro.get('descricao_item', '')}",
        f"Quantidade inspecionada: {quantidade_inspecionada}",
        f"Horário de login do operador: {registro['horario_login']}",
        "Horários de detecção de ausência:",
    ]
    if registro["ausencias"]:
        linhas += [f"  {a['horario']}" for a in registro["ausencias"]]
    else:
        linhas.append("  Nenhuma ausência detectada")

    PASTA_RELATORIOS.mkdir(parents=True, exist_ok=True)
    # lote == ordem (ver consultar_ordem), então o nome leva só a ordem
    caminho = PASTA_RELATORIOS / f"ordem_{registro['ordem']}_{carimbo}.txt"
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return caminho


def finalizar_ordem(numero_ordem: str) -> Path:
    """Gera o txt e fecha o registro da ordem."""
    numero_ordem = _validar_numero_ordem(numero_ordem)
    registro = _ler_registro(numero_ordem)
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho_txt = gerar_txt(registro, carimbo)
    # guarda o registro fechado (histórico) e libera a ordem para um novo 'iniciar'
    origem = _caminho_registro(numero_ordem)
    origem.replace(origem.with_name(f"ordem_{numero_ordem}_{carimbo}.finalizado.json"))
    return caminho_txt


# ------------------------------------------------------------
# Ponto de entrada para o LabVIEW
# ------------------------------------------------------------
def main():
    args = sys.argv[1:]
    comando = args[0].lower() if args else ""

    try:
        if comando == "iniciar" and len(args) == 2:
            r = iniciar_ordem(args[1])
            print(f"{r['lote']};{r['quantidade_lote']};OK;")
        elif comando == "registrar" and len(args) == 3:
            r = registrar_bandeja(args[1], int(args[2]))
            print(f"{_total_vazias(r)};OK;")
        elif comando == "finalizar" and len(args) == 2:
            print(f"{finalizar_ordem(args[1])};OK;")
        else:
            print("Uso: python erp_doc.py iniciar <ordem> | registrar <ordem> <vazias> | finalizar <ordem>",
                  file=sys.stderr)
            falha = {"iniciar": ";0;ERRADO;", "registrar": "0;ERRADO;"}.get(comando, ";ERRADO;")
            print(falha)
    except Exception as erro:
        print(f"Falha em '{comando}': {erro}", file=sys.stderr)
        falha = {"iniciar": ";0;ERRADO;", "registrar": "0;ERRADO;"}.get(comando, ";ERRADO;")
        print(falha)


if __name__ == "__main__":
    main()
