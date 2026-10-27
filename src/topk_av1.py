from av2_types import Candidate, Selection, make_selection, validate
from sorting import merge_sort


def solve(candidates: list[Candidate], budget: int, k: int = 5) -> Selection:
    """Baseline da AV1: ordena os candidatos por score decrescente com
    merge_sort e seleciona os k primeiros, ignorando o orçamento budget."""
    validate(candidates, budget)
    by_id = {c.doc_id: c for c in candidates}
    pairs = [(c.doc_id, c.value) for c in candidates]
    sorted_pairs, comparisons = merge_sort(pairs)
    chosen = [by_id[doc_id] for doc_id, _ in sorted_pairs[:k]]
    return make_selection(chosen, comparisons)

