from typing import TYPE_CHECKING

from av2_types import Candidate
from inverted_index import indexed_search
from sorting import merge_sort

if TYPE_CHECKING:
    from pipeline import Corpus


def build_candidates(
    corpus: "Corpus", query_tokens: list[str], m: int = 200
) -> list[Candidate]:
    """Recupera os documentos candidatos via índice invertido (BM25),
    ordena por score decrescente com merge_sort (desempate por doc_id
    crescente) e devolve até os m primeiros como objetos Candidate."""
    if corpus.inverted_index is None:
        raise ValueError(
            "corpus.inverted_index e None -- chame build_corpus_index(corpus) "
            "antes de construir os candidatos."
        )
    pairs, _ = indexed_search(query_tokens, corpus.inverted_index, corpus.bm25_stats)
    sorted_pairs, _ = merge_sort(pairs)
    positive_pairs = [
        (doc_id, score)
        for doc_id, score in sorted_pairs
        if score > 0 and corpus.bm25_stats.document_lengths[doc_id] > 0
    ]
    top_m = positive_pairs[:m]
    return [
        Candidate(
            doc_id=doc_id,
            value=score,
            cost=corpus.bm25_stats.document_lengths[doc_id],
        )
        for doc_id, score in top_m
    ]
