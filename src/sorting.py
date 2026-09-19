from collections.abc import Sequence

DocScorePair = tuple[int, float]


def comes_after(pair_a: DocScorePair, pair_b: DocScorePair) -> bool:
    """Define o critério de ordenação usado por insertion_sort e
    merge_sort: score decrescente, com desempate por doc_id crescente.
    Retorna True se pair_a deve ficar depois de pair_b na lista
    ordenada."""
    doc_a, score_a = pair_a
    doc_b, score_b = pair_b
    if score_a != score_b:
        return score_a < score_b
    return doc_a > doc_b


def insertion_sort(pairs: Sequence[DocScorePair]) -> tuple[list[DocScorePair], int]:
    """Ordena uma lista de pares (doc_id, score) por score decrescente,
    usando o critério de desempate de comes_after. Devolve a lista
    ordenada e o número de comparações realizadas."""
    result = list(pairs)
    comparisons = 0
    for j in range(1, len(result)):
        key = result[j]
        i = j - 1
        while i >= 0:
            comparisons += 1
            if comes_after(result[i], key):
                result[i + 1] = result[i]
                i -= 1
            else:
                break
        result[i + 1] = key
    return result, comparisons


def merge_sort(pairs: Sequence[DocScorePair]) -> tuple[list[DocScorePair], int]:
    """Ordena uma lista de pares (doc_id, score) por score decrescente,
    dividindo a lista ao meio recursivamente e depois combinando as
    partes já ordenadas. Usa o mesmo critério de desempate de
    comes_after. Devolve a lista ordenada e o número de comparações
    realizadas."""
    result = list(pairs)
    comparisons = 0

    def _merge(start: int, mid: int, end: int) -> None:
        nonlocal comparisons
        left = result[start:mid]
        right = result[mid:end]
        i = j = 0
        k = start
        while i < len(left) and j < len(right):
            comparisons += 1
            if comes_after(left[i], right[j]):
                result[k] = right[j]
                j += 1
            else:
                result[k] = left[i]
                i += 1
            k += 1
        while i < len(left):
            result[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            result[k] = right[j]
            j += 1
            k += 1

    def _merge_sort(start: int, end: int) -> None:
        if end - start <= 1:
            return
        mid = (start + end) // 2
        _merge_sort(start, mid)
        _merge_sort(mid, end)
        _merge(start, mid, end)

    _merge_sort(0, len(result))
    return result, comparisons
