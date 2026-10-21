# Contrato fixo da AV2 (Trilha A, mochila 0/1). Copiar sem alterar.
from typing import NamedTuple


class Candidate(NamedTuple):
    doc_id: int    # identificador do documento (único)
    value: float   # score BM25 (>= 0)
    cost: int      # len(tokens) do documento (> 0)


class Selection(NamedTuple):
    doc_ids: list[int]   # em ordem crescente
    total_value: float
    total_cost: int
    ops: int             # contagem de operações (regra de cada método)


def validate(candidates: list[Candidate], budget: int) -> None:
    # Todo solve chama esta função antes de qualquer outra coisa.
    if budget < 0:
        raise ValueError('orcamento negativo')
    seen: set[int] = set()
    for c in candidates:
        if c.cost <= 0:
            raise ValueError('custo deve ser positivo')
        if c.value < 0:
            raise ValueError('valor negativo')
        if c.doc_id in seen:
            raise ValueError('doc_id repetido')
        seen.add(c.doc_id)


def make_selection(chosen: list[Candidate], ops: int) -> Selection:
    ordered = sorted(chosen, key=lambda c: c.doc_id)
    return Selection(
        [c.doc_id for c in ordered],
        sum(c.value for c in ordered),
        sum(c.cost for c in ordered),
        ops,
    )
