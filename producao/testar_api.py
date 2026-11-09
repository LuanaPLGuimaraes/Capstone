"""
Teste manual da API do ERP: digite o número de uma ordem e veja a resposta.

Só consulta o ERP. Não cria registro de ordem nem gera relatório.

Uso: botão "Run Python File" do VS Code ou
    python producao/testar_api.py
"""
from erp_doc import consultar_ordem


def main():
    while True:
        numero = input("Número da ordem (Enter para sair): ").strip()
        if not numero:
            break
        try:
            ordem = consultar_ordem(numero)
        except Exception as erro:
            print(f"  FALHOU: {erro}\n")
            continue
        print("  OK, resposta do ERP:")
        for campo, valor in ordem.items():
            print(f"    {campo}: {valor}")
        print()


if __name__ == "__main__":
    main()
