from av2_types import Candidate, Selection, make_selection, validate


def solve(candidates: list[Candidate], budget: int) -> Selection:
    """Seleciona um subconjunto de valor máximo dentro do orçamento."""
    validate(candidates, budget)

    n = len(candidates)
    dp: list[list[float]] = [[0.0] * (budget + 1) for _ in range(n + 1)]

    for i, candidate in enumerate(candidates, start=1):
        for w in range(budget + 1):
            dp[i][w] = dp[i - 1][w]
            if candidate.cost <= w:
                dp[i][w] = max(
                    dp[i][w],
                    dp[i - 1][w - candidate.cost] + candidate.value,
                )

    chosen: list[Candidate] = []
    remaining_budget = budget
    for i in range(n, 0, -1):
        if dp[i][remaining_budget] != dp[i - 1][remaining_budget]:
            candidate = candidates[i - 1]
            chosen.append(candidate)
            remaining_budget -= candidate.cost

    return make_selection(chosen, n * (budget + 1))
