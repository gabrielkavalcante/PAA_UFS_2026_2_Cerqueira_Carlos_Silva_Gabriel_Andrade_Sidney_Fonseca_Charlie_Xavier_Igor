from bm25 import BM25Stats, idf, tf

# posting = (doc_id, frequência do termo no documento)
PostingList = list[tuple[int, int]]
InvertedIndex = dict[str, PostingList]


def build_inverted_index(bm25_stats: BM25Stats) -> InvertedIndex:
    """Constrói um índice que associa cada termo à sua posting list,
    contendo os documentos em que o termo aparece e sua frequência.
    Executado uma única vez para as configurações C2 e C3."""
    postings_by_term: InvertedIndex = {}
    for doc_id, term_frequency in enumerate(bm25_stats.term_frequency_by_document):
        for term, frequency in term_frequency.items():
            postings_by_term.setdefault(term, []).append((doc_id, frequency))
    return postings_by_term


def indexed_search(
    query_tokens: list[str],
    inverted_index: InvertedIndex,
    bm25_stats: BM25Stats,
) -> tuple[list[tuple[int, float]], int]:
    """Usada pelas configurações C2 e C3: consulta diretamente no índice
    invertido a posting list de cada termo e acumula o score somente nos
    documentos candidatos, isto é, nos documentos que contêm pelo menos
    um termo da consulta.

    Devolve a lista de pares (doc_id, score) apenas dos candidatos e o
    número de operações realizadas durante a consulta."""
    n_docs = len(bm25_stats.term_frequency_by_document)
    scores: dict[int, float] = {}
    n_operations = 0

    for term in query_tokens:
        posting_list = inverted_index.get(term)
        n_operations += 1
        if posting_list is None:
            continue  # termo fora do vocabulário

        idf_value = idf(len(posting_list), n_docs)
        for doc_id, frequency in posting_list:
            n_operations += 1
            contribution = idf_value * tf(
                frequency,
                bm25_stats.document_lengths[doc_id],
                bm25_stats.average_document_length,
            )
            scores[doc_id] = scores.get(doc_id, 0.0) + contribution

    return list(scores.items()), n_operations
