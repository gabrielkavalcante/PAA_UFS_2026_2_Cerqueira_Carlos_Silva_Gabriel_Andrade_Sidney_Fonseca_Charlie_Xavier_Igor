import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import pytest
from av2_types import Candidate, Selection
from fixtures import F1, F2, F3, F4, F5, F6, F7, F8, F9

import brute_force
import topk_av1


def test_t05_brute_force():
    # F7: devolve [1, 3] com valor 13.0 e custo 9 (ops = 2^3 = 8)
    budget_f7, cands_f7 = F7
    res_f7 = brute_force.solve(cands_f7, budget_f7)
    assert res_f7.doc_ids == [1, 3]
    assert res_f7.total_value == pytest.approx(13.0, abs=1e-9)
    assert res_f7.total_cost == 9
    assert res_f7.ops == 8

    # F9: devolve [2, 3, 4] com valor 9.1 e custo 10 (ops = 2^4 = 16)
    budget_f9, cands_f9 = F9
    res_f9 = brute_force.solve(cands_f9, budget_f9)
    assert res_f9.doc_ids == [2, 3, 4]
    assert res_f9.total_value == pytest.approx(9.1, abs=1e-9)
    assert res_f9.total_cost == 10
    assert res_f9.ops == 16

    # F8: doc_id repetido levanta ValueError
    budget_f8, cands_f8 = F8
    with pytest.raises(ValueError):
        brute_force.solve(cands_f8, budget_f8)


def test_t06_topk_av1():
    # F4: devolve [1, 2] com custo 11 (ultrapassa W = 4)
    budget_f4, cands_f4 = F4
    res_f4 = topk_av1.solve(cands_f4, budget_f4)
    assert res_f4.doc_ids == [1, 2]
    assert res_f4.total_value == pytest.approx(17.0, abs=1e-9)
    assert res_f4.total_cost == 11
    assert res_f4.total_cost > budget_f4
    assert res_f4.ops == 1

    # F5: sem candidatos -> seleção vazia
    budget_f5, cands_f5 = F5
    res_f5 = topk_av1.solve(cands_f5, budget_f5)
    assert res_f5.doc_ids == []
    assert res_f5.total_value == pytest.approx(0.0, abs=1e-9)
    assert res_f5.total_cost == 0
    assert res_f5.ops == 0

