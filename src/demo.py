"""
Demonstracao para o checkpoint de 10/09.
Configuracao C1 do plano: busca linear + Insertion Sort.
Rodar: python demo.py
(articles.csv precisa estar na mesma pasta -- baixar do Kaggle)
"""

from bm25 import busca_linear, construir_estatisticas
from ingest import carregar_e_processar_corpus, limpar_texto, tokenizar
from sorting import insertion_sort

CAMINHO_CSV: str = "articles.csv"
N_AMOSTRA: int = 5000
K: int = 5
CONSULTA: str = "impeachment de Dilma Rousseff"


def main() -> None:
    df = carregar_e_processar_corpus(CAMINHO_CSV, n_amostra=N_AMOSTRA)
    lista_tokens = df["tokens"].tolist()

    freq_por_doc, tamanhos, avgdl, df_termo = construir_estatisticas(lista_tokens)

    query_tokens = tokenizar(limpar_texto(CONSULTA))
    print(f"\nConsulta: '{CONSULTA}'")
    print(f"Tokens normalizados: {query_tokens}\n")

    resultados = busca_linear(
        query_tokens, lista_tokens, freq_por_doc, tamanhos, avgdl, df_termo
    )
    resultados_ordenados, comparacoes = insertion_sort(resultados)

    print(
        f"Top-{K} de {len(df)} documentos ({comparacoes} comparacoes na ordenacao):\n"
    )
    for doc_idx, score in resultados_ordenados[:K]:
        linha = df.iloc[doc_idx]
        print(f"[score {score:.3f}] {linha['title']}  ({linha['date']})")


if __name__ == "__main__":
    main()
