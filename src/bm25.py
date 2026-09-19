import math
from collections import Counter
from typing import NamedTuple

K1: float = 1.2
B: float = 0.75


class BM25Stats(NamedTuple):
    # Número de vezes em que um termo t aparece em um documento d
    term_frequency_by_document: list[Counter[str]]
    # Número de documentos em que um termo t aparece
    document_frequency_by_term: Counter[str]
    # document_lengths[d] = tamanho do documento d
    document_lengths: list[int]
    # Tamanho médio dos documentos
    average_document_length: float


def build_bm25_stats(
    documents_tokens: list[list[str]],
) -> BM25Stats:
    """Recebe uma lista de listas de tokens (um item por documento) e
    devolve o que o BM25 precisa: frequência dos termos em cada
    documento, tamanho de cada documento, tamanho médio (avgdl) e em
    quantos documentos cada termo aparece (para o cálculo do IDF)."""

    term_freq: list[Counter[str]] = [Counter(tokens) for tokens in documents_tokens]
    doc_lens = [len(tokens) for tokens in documents_tokens]
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 0.0
    df: Counter[str] = Counter()
    for doc in term_freq:
        for term in doc:
            df[term] += 1

    return BM25Stats(
        term_frequency_by_document=term_freq,
        document_lengths=doc_lens,
        average_document_length=avgdl,
        document_frequency_by_term=df,
    )


def idf(doc_count_by_term: int, n_docs: int) -> float:
    """Calcula o IDF de um termo, dada a contagem de documentos em que ele aparece."""
    return math.log(1 + (n_docs - doc_count_by_term + 0.5) / (doc_count_by_term + 0.5))


def tf(freq: int, doc_len: int, avgdl: float) -> float:
    """Calcula o TF de um termo, dada a frequência do termo no documento."""
    return freq / (freq + K1 * (1 - B + B * doc_len / avgdl if avgdl > 0 else 0.0))


def score_bm25(
    query_tokens: list[str],
    doc_idx: int,
    n_docs: int,
    bm25_stats: BM25Stats,
) -> tuple[float, int]:
    """Calcula o score BM25 de um documento em relação à consulta.
    Além do score, devolve quantos termos da consulta foram verificados
    contra o documento -- essa contagem é usada pela busca linear
    (configuração C1) para medir o custo da busca."""
    term_frequency = bm25_stats.term_frequency_by_document[doc_idx]
    document_frequency = bm25_stats.document_frequency_by_term
    doc_len = bm25_stats.document_lengths[doc_idx]
    avgdl = bm25_stats.average_document_length

    score = 0.0
    n_operations = 0

    for term in query_tokens:
        n_operations += 1

        freq = term_frequency.get(term, 0)
        if freq == 0:
            continue

        document_count = document_frequency.get(term, 0)
        score += idf(document_count, n_docs) * tf(freq, doc_len, avgdl)

    return score, n_operations
