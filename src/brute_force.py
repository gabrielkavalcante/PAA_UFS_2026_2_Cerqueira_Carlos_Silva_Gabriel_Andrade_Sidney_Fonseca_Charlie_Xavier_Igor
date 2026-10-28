from av2_types import Candidate, Selection, make_selection, validate


def solve(candidates: list[Candidate], budget: int) -> Selection:
    """Resolve a mochila 0/1 por enumeração exaustiva dos 2^n subconjuntos.
    Aceita no máximo 20 candidatos; acima disso levanta ValueError."""
    validate(candidates, budget)
    n = len(candidates)
    if n > 20:
        raise ValueError("brute_force aceita no maximo 20 candidatos")

    best_value = -1.0
    best_subset: list[Candidate] = []

    for mask in range(1 << n):
        subset: list[Candidate] = []
        total_cost = 0
        total_value = 0.0
        for i in range(n):
            if (mask >> i) & 1:
                c = candidates[i]
                total_cost += c.cost
                total_value += c.value
                subset.append(c)

        if total_cost <= budget and total_value > best_value:
            best_value = total_value
            best_subset = subset

    return make_selection(best_subset, 2**n)

