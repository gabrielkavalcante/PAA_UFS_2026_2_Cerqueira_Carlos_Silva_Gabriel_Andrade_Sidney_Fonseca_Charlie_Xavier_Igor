import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import pytest
from av2_types import Candidate, Selection
from fixtures import F1, F2, F3, F4, F5, F6, F7, F8, F9

from knapsack_greedy import solve_ratio, solve_ratio_plus, solve_score

SOLVERS = [solve_score, solve_ratio, solve_ratio_plus]


def test_t03_ratio_erra_e_ratio_plus_corrige():
    # F1 é adversarial: o item 2 tem a melhor razão (2,0 contra 1,0),
    # mas ocupa 1 token e impede o item 1, que sozinho vale 10,0.
    budget, candidatos = F1

    ratio = solve_ratio(candidatos, budget)
    assert ratio.doc_ids == [2]
    assert ratio.total_value == pytest.approx(2.0, abs=1e-9)
    assert ratio.total_cost <= budget

    plus = solve_ratio_plus(candidatos, budget)
    assert plus.doc_ids == [1]
    assert plus.total_value == pytest.approx(10.0, abs=1e-9)
    assert plus.total_cost <= budget


@pytest.mark.parametrize('solve', SOLVERS, ids=lambda f: f.__name__)
def test_t04_empate_deterministico_e_tudo_cabe(solve):
    # F2: três documentos iguais (5,0 e 5 tokens) e W = 10; só cabem dois.
    budget, candidatos = F2
    primeira = solve(candidatos, budget)
    segunda = solve(candidatos, budget)
    assert primeira.total_value == pytest.approx(10.0, abs=1e-9)
    assert len(primeira.doc_ids) == 2
    assert primeira.total_cost <= budget
    assert segunda == primeira

    # F3: a soma dos custos (60) cabe em W = 100, então entram os três.
    budget, candidatos = F3
    tudo = solve(candidatos, budget)
    assert tudo.doc_ids == [1, 2, 3]
    assert tudo.total_value == pytest.approx(12.0, abs=1e-9)
    assert tudo.total_cost == 60
