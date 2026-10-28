import sys
from itertools import combinations
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from fixtures import F1, F2, F3, F4, F5, F6, F7, F8, F9

from av2_types import Candidate, Selection
from knapsack_dp import solve


def _solve_fixture(fixture: tuple[int, list[Candidate]]) -> Selection:
    budget, candidates = fixture
    return solve(candidates, budget)


def test_t01_instancias_principais() -> None:
    resultado_f1 = _solve_fixture(F1)
    assert resultado_f1 == Selection([1], 10.0, 10, 22)

    resultado_f7 = _solve_fixture(F7)
    assert resultado_f7 == Selection([1, 3], 13.0, 9, 33)

    resultado_f4 = _solve_fixture(F4)
    assert resultado_f4 == Selection([], 0.0, 0, 10)

    resultado_f6 = _solve_fixture(F6)
    assert resultado_f6 == Selection([], 0.0, 0, 1)


def test_t02_f9_refuta_preservacao_do_documento_mais_relevante() -> None:
    resultado = _solve_fixture(F9)

    assert resultado.doc_ids == [2, 3, 4]
    assert resultado.total_value == pytest.approx(9.1)
    assert resultado.total_cost == 10
    assert resultado.ops == 44


def test_corpus_vazio() -> None:
    resultado = _solve_fixture(F5)

    assert resultado == Selection([], 0.0, 0, 0)


def test_empate_escolhe_solucao_otima_deterministica() -> None:
    resultado = _solve_fixture(F2)

    assert resultado == Selection([1, 2], 10.0, 10, 33)


def test_todos_os_itens_cabem() -> None:
    resultado = _solve_fixture(F3)

    assert resultado == Selection([1, 2, 3], 12.0, 60, 303)


def test_entrada_grande_respeita_orcamento_e_conta_operacoes() -> None:
    candidates = [Candidate(index, float(index), 10) for index in range(1, 201)]
    budget = 1000

    resultado = solve(candidates, budget)

    assert resultado.total_cost <= budget
    assert resultado.ops == len(candidates) * (budget + 1)
    assert resultado.doc_ids == sorted(set(resultado.doc_ids))
    chosen = [c for c in candidates if c.doc_id in resultado.doc_ids]
    assert resultado.total_cost == sum(c.cost for c in chosen)
    assert resultado.total_value == pytest.approx(sum(c.value for c in chosen))
    # Custos iguais: o ótimo contém os 100 maiores valores, de 101 a 200.
    assert resultado.doc_ids == list(range(101, 201))
    assert resultado.total_value == 15050.0
    assert resultado.total_cost == budget


def test_doc_id_duplicado_e_rejeitado() -> None:
    with pytest.raises(ValueError, match="doc_id repetido"):
        _solve_fixture(F8)


def test_orcamento_negativo_e_rejeitado() -> None:
    with pytest.raises(ValueError, match="orcamento negativo"):
        solve([], -1)


@pytest.mark.parametrize("cost", [0, -1])
def test_custo_nao_positivo_e_rejeitado(cost: int) -> None:
    with pytest.raises(ValueError, match="custo deve ser positivo"):
        solve([Candidate(1, 1.0, cost)], 2)


def test_valor_negativo_e_rejeitado() -> None:
    with pytest.raises(ValueError, match="valor negativo"):
        solve([Candidate(1, -1.0, 1)], 2)


def test_empate_preserva_ordem_recebida_sem_alterar_entrada() -> None:
    candidates = [Candidate(9, 5.0, 2), Candidate(1, 5.0, 2)]
    original = candidates.copy()

    assert solve(candidates, 2) == Selection([9], 5.0, 2, 6)
    assert solve(candidates, 4) == Selection([1, 9], 10.0, 4, 10)
    assert candidates == original


def test_item_nao_pode_ser_reutilizado() -> None:
    assert solve([Candidate(1, 3.0, 2)], 6) == Selection([1], 3.0, 2, 7)


def test_valor_zero_nao_e_incluido_em_empate() -> None:
    assert solve([Candidate(1, 0.0, 1)], 1) == Selection([], 0.0, 0, 2)


def test_otimalidade_contra_enumeracao_exaustiva() -> None:
    rng = Random(42)
    for _ in range(100):
        candidates = [
            Candidate(i, rng.randrange(41) / 4, rng.randrange(1, 11))
            for i in range(rng.randrange(9))
        ]
        rng.shuffle(candidates)
        budget = rng.randrange(21)
        # Valores múltiplos de 1/4 evitam arredondamentos na comparação do oráculo.
        optimal_value = max(
            sum(c.value for c in subset)
            for size in range(len(candidates) + 1)
            for subset in combinations(candidates, size)
            if sum(c.cost for c in subset) <= budget
        )

        result = solve(candidates, budget)
        chosen = [c for c in candidates if c.doc_id in result.doc_ids]
        assert result.total_value == optimal_value
        assert result.total_cost == sum(c.cost for c in chosen) <= budget
        assert result.total_value == sum(c.value for c in chosen)
        assert result.doc_ids == sorted(set(result.doc_ids))
        assert result.ops == len(candidates) * (budget + 1)
