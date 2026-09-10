from collections.abc import Sequence

ParDocScore = tuple[int, float]


def vem_depois(par_a: ParDocScore, par_b: ParDocScore) -> bool:
    """Define a ordem total do plano (secao 2.1): score decrescente,
    desempate por doc_id crescente. Retorna True se par_a deve ficar
    DEPOIS de par_b na lista ordenada."""
    doc_a, score_a = par_a
    doc_b, score_b = par_b
    if score_a != score_b:
        return score_a < score_b
    return doc_a > doc_b


def insertion_sort(pares: Sequence[ParDocScore]) -> tuple[list[ParDocScore], int]:
    """Ordena uma lista de (doc_id, score) por score decrescente.
    Retorna a lista ordenada e o numero de comparacoes feitas --
    usado depois na Parte B do plano (analise assintotica)."""
    A = list(pares)
    comparacoes = 0
    for j in range(1, len(A)):
        chave = A[j]
        i = j - 1
        while i >= 0:
            comparacoes += 1
            if vem_depois(A[i], chave):
                A[i + 1] = A[i]
                i -= 1
            else:
                break
        A[i + 1] = chave
    return A, comparacoes
