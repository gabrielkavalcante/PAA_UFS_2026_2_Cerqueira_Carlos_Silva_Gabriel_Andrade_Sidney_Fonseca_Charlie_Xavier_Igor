from bm25 import BM25Stats, score_bm25


def linear_search(
    query_tokens: list[str],
    bm25_stats: BM25Stats,
) -> tuple[list[tuple[int, float]], int]:
    """Configuração C1: percorre todos os documentos do corpus e calcula
    o score BM25 de cada um contra a consulta, mesmo quando o documento
    não compartilha nenhum termo com ela -- nenhum documento é
    descartado antes de ser examinado.

    Devolve a lista de pares (doc_id, score) para todos os documentos e
    o número total de verificações (documento, termo) realizadas."""
    n_docs = len(bm25_stats.term_frequency_by_document)
    results: list[tuple[int, float]] = []
    n_operations = 0

    for doc_idx in range(n_docs):
        score, operations = score_bm25(query_tokens, doc_idx, n_docs, bm25_stats)
        n_operations += operations
        results.append((doc_idx, score))

    return results, n_operations
