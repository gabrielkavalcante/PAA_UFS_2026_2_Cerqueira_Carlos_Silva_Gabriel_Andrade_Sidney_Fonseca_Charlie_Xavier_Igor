from av2_types import Candidate, Selection, make_selection, validate
from sorting import merge_sort


def _greedy_by_key(
    candidates: list[Candidate], budget: int, keys: dict[int, float]
) -> tuple[list[Candidate], int]:
    """Ordena os candidatos pela chave decrescente com o merge_sort da
    AV1 (desempate por doc_id crescente) e percorre a lista uma vez,
    pegando cada item que ainda cabe no que sobra do orçamento. Devolve
    os itens escolhidos e ops = comparações do merge_sort + n."""
    by_id = {c.doc_id: c for c in candidates}
    ordered, comparisons = merge_sort([(c.doc_id, keys[c.doc_id]) for c in candidates])
    chosen: list[Candidate] = []
    remaining = budget
    for doc_id, _ in ordered:
        item = by_id[doc_id]
        if item.cost <= remaining:
            chosen.append(item)
            remaining -= item.cost
    return chosen, comparisons + len(candidates)


def solve_score(candidates: list[Candidate], budget: int) -> Selection:
    """Guloso pelo maior score: percorre os candidatos por value
    decrescente e pega os que cabem. Sem garantia de aproximação."""
    validate(candidates, budget)
    keys = {c.doc_id: c.value for c in candidates}
    chosen, ops = _greedy_by_key(candidates, budget, keys)
    return make_selection(chosen, ops)


def solve_ratio(candidates: list[Candidate], budget: int) -> Selection:
    """Guloso pela razão value/cost: percorre os candidatos por razão
    decrescente e pega os que cabem. Ótimo na mochila fracionária, mas
    sem garantia na 0/1 (ver F1)."""
    validate(candidates, budget)
    keys = {c.doc_id: c.value / c.cost for c in candidates}
    chosen, ops = _greedy_by_key(candidates, budget, keys)
    return make_selection(chosen, ops)


def solve_ratio_plus(candidates: list[Candidate], budget: int) -> Selection:
    """Compara o resultado de solve_ratio com o melhor item isolado que
    cabe (maior value com cost <= budget, desempate por doc_id menor) e
    devolve o de maior valor; em empate, fica com o de solve_ratio.
    Garante pelo menos metade do valor ótimo."""
    validate(candidates, budget)
    keys = {c.doc_id: c.value / c.cost for c in candidates}
    chosen, ops = _greedy_by_key(candidates, budget, keys)
    ratio_value = sum(c.value for c in chosen)

    best: Candidate | None = None
    for c in candidates:
        if c.cost <= budget and (
            best is None
            or c.value > best.value
            or (c.value == best.value and c.doc_id < best.doc_id)
        ):
            best = c

    if best is not None and best.value > ratio_value:
        return make_selection([best], ops)
    return make_selection(chosen, ops)
