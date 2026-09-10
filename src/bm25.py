import math
from collections import Counter

K1: float = 1.2
B: float = 0.75

FrequenciasPorDoc = list[Counter[str]]


def construir_estatisticas(
    lista_tokens: list[list[str]],
) -> tuple[FrequenciasPorDoc, list[int], float, Counter[str]]:
    """Recebe uma lista de listas de tokens (um item por documento) e
    devolve o que o BM25 precisa: frequencia dos termos em cada
    documento, tamanho de cada documento, tamanho medio (avgdl) e em
    quantos documentos cada termo aparece (para o IDF)."""
    freq_por_doc: FrequenciasPorDoc = [Counter(tokens) for tokens in lista_tokens]
    tamanhos = [len(tokens) for tokens in lista_tokens]
    avgdl = sum(tamanhos) / len(tamanhos) if tamanhos else 0.0

    df_termo: Counter[str] = Counter()
    for freq in freq_por_doc:
        for termo in freq:
            df_termo[termo] += 1

    return freq_por_doc, tamanhos, avgdl, df_termo


def idf(termo: str, df_termo: Counter[str], n_docs: int) -> float:
    # variante do Lucene (com +1 interno) -- ver plano, secao 2.1
    n_t = df_termo.get(termo, 0)
    return math.log((n_docs - n_t + 0.5) / (n_t + 0.5) + 1)


def score_bm25(
    query_tokens: list[str],
    doc_idx: int,
    freq_por_doc: FrequenciasPorDoc,
    tamanhos: list[int],
    avgdl: float,
    df_termo: Counter[str],
    n_docs: int,
) -> float:
    score = 0.0
    freq_doc = freq_por_doc[doc_idx]
    tamanho_doc = tamanhos[doc_idx]
    for termo in query_tokens:
        f = freq_doc.get(termo, 0)
        if f == 0:
            continue
        numerador = f * (K1 + 1)
        denominador = f + K1 * (1 - B + B * tamanho_doc / avgdl)
        score += idf(termo, df_termo, n_docs) * (numerador / denominador)
    return score


def busca_linear(
    query_tokens: list[str],
    lista_tokens: list[list[str]],
    freq_por_doc: FrequenciasPorDoc,
    tamanhos: list[int],
    avgdl: float,
    df_termo: Counter[str],
) -> list[tuple[int, float]]:
    """Configuracao 1: varre todos os documentos e calcula
    o score BM25 de cada um contra a consulta."""
    n_docs = len(lista_tokens)
    return [
        (
            doc_idx,
            score_bm25(
                query_tokens, doc_idx, freq_por_doc, tamanhos, avgdl, df_termo, n_docs
            ),
        )
        for doc_idx in range(n_docs)
    ]
