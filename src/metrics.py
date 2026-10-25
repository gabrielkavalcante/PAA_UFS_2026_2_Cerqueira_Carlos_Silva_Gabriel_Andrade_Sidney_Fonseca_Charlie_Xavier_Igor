"""Pacote 4 — Avaliação de métricas de qualidade do contexto selecionado."""

from itertools import combinations
from typing import Any
from av2_types import Selection


def _doc_token_set(documents_tokens: Any, doc_id: int) -> set[str]:
    """Devolve o conjunto de tokens distintos do documento `doc_id`."""
    if isinstance(documents_tokens, dict):
        return set(documents_tokens.get(doc_id, []))
    if 0 <= doc_id < len(documents_tokens):
        return set(documents_tokens[doc_id])
    return set()


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Índice de Jaccard entre dois conjuntos de tokens."""
    union_size = len(set_a | set_b)
    if union_size == 0:
        return 0.0
    return len(set_a & set_b) / union_size


def evaluate(
    selection: Selection,
    budget: int,
    query_tokens: list[str],
    documents_tokens: Any,
) -> dict[str, Any]:
    """Calcula as métricas do contexto selecionado para um orçamento `budget`.

    Retorna dicionário com:
      - n_docs_sel: número de documentos escolhidos
      - tokens_used: selection.total_cost
      - budget_ok: True se tokens_used <= budget
      - total_score: selection.total_value
      - coverage: termos distintos da consulta na união dos escolhidos / termos distintos da consulta
      - redundancy: média do Jaccard entre os conjuntos de tokens de cada par de documentos escolhidos
    """
    doc_ids = selection.doc_ids
    n_docs_sel = len(doc_ids)
    tokens_used = int(selection.total_cost)
    budget_ok = bool(tokens_used <= budget)
    total_score = float(selection.total_value)

    doc_sets = [_doc_token_set(documents_tokens, doc_id) for doc_id in doc_ids]
    query_set = set(query_tokens)

    # Cobertura: 0.0 se a consulta ou a seleção for vazia
    if not query_set or n_docs_sel == 0:
        coverage = 0.0
    else:
        union_tokens = set().union(*doc_sets)
        coverage = len(query_set & union_tokens) / len(query_set)

    # Redundância: média do Jaccard entre todos os pares (0.0 com menos de 2 documentos)
    if n_docs_sel < 2:
        redundancy = 0.0
    else:
        pair_scores = [_jaccard(a, b) for a, b in combinations(doc_sets, 2)]
        redundancy = sum(pair_scores) / len(pair_scores)

    return {
        "n_docs_sel": n_docs_sel,
        "tokens_used": tokens_used,
        "budget_ok": budget_ok,
        "total_score": total_score,
        "coverage": coverage,
        "redundancy": redundancy,
    }