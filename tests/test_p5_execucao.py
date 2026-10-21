import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import pytest
from av2_types import Candidate, Selection
from fixtures import F1, F2, F3, F4, F5, F6, F7, F8, F9
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'experiments'))

import csv

import run_av2
from knapsack_dp import solve as solve_dp
from knapsack_greedy import solve_ratio_plus


def test_t09_execucao_curta(tmp_path):
    # Grade mínima: N = 1.000, consulta q1, W = 500, 1 repetição.
    codigo = run_av2.main([
        '--n', '1000', '--queries', '1', '--w', '500', '--reps', '1',
        '--out-dir', str(tmp_path), '--skip-validation',
    ])
    assert codigo == 0
    with open(tmp_path / 'av2_runs.csv', newline='', encoding='utf-8') as f:
        leitor = csv.DictReader(f)
        assert leitor.fieldnames == run_av2.COLUMNS
        linhas = list(leitor)
    verificados = [l for l in linhas if l['method'] in
                   ('dp', 'greedy_score', 'greedy_ratio', 'greedy_ratio_plus')]
    assert len(verificados) == 4
    assert all(l['budget_ok'] == '1' for l in verificados)


def test_t10_entrada_grande():
    # 200 itens sintéticos (seed 42), custos de 10 a 300, W = 2.000.
    _, candidatos = run_av2.synthetic_instance(42, 200, cost_range=(10, 300))
    orcamento = 2000
    dp = solve_dp(candidatos, orcamento)
    plus = solve_ratio_plus(candidatos, orcamento)
    assert dp.total_value >= plus.total_value - 1e-9
    assert plus.total_value >= dp.total_value / 2 - 1e-9
    assert dp.total_cost <= orcamento
    assert plus.total_cost <= orcamento
