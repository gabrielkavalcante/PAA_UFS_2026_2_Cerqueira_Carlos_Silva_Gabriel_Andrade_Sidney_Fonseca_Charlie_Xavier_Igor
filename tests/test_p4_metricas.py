import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import pytest
from av2_types import Candidate, Selection
from fixtures import F1, F2, F3, F4, F5, F6, F7, F8, F9

from metrics import evaluate

TOKENS_FIXTURE = [[], ['a', 'b'], ['b', 'c']]
QUERY_FIXTURE = ['a', 'b', 'z']


def test_t07_selecao_com_dois_documentos():
    # Seleção [1, 2] com custo 4 e valor 10.0
    sel = Selection(doc_ids=[1, 2], total_value=10.0, total_cost=4, ops=5)

    # Caso 1: budget = 10 (cabe no orçamento)
    res_ok = evaluate(sel, 10, QUERY_FIXTURE, TOKENS_FIXTURE)
    assert res_ok['n_docs_sel'] == 2
    assert res_ok['tokens_used'] == 4
    assert res_ok['budget_ok'] is True
    assert res_ok['total_score'] == pytest.approx(10.0, abs=1e-9)
    assert res_ok['coverage'] == pytest.approx(2 / 3, abs=1e-9)
    assert res_ok['redundancy'] == pytest.approx(1 / 3, abs=1e-9)

    # Caso 2: budget = 3 (estoura o orçamento)
    res_estouro = evaluate(sel, 3, QUERY_FIXTURE, TOKENS_FIXTURE)
    assert res_estouro['budget_ok'] is False
    assert res_estouro['tokens_used'] == 4
    assert res_estouro['n_docs_sel'] == 2


def test_t08_selecao_vazia():
    sel_vazia = Selection(doc_ids=[], total_value=0.0, total_cost=0, ops=0)
    res = evaluate(sel_vazia, 10, QUERY_FIXTURE, TOKENS_FIXTURE)

    assert res['n_docs_sel'] == 0
    assert res['tokens_used'] == 0
    assert res['budget_ok'] is True
    assert res['total_score'] == pytest.approx(0.0, abs=1e-9)
    assert res['coverage'] == pytest.approx(0.0, abs=1e-9)
    assert res['redundancy'] == pytest.approx(0.0, abs=1e-9)