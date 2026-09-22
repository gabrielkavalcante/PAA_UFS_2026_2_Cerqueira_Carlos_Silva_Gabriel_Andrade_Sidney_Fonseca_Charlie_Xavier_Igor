"""Calcula o Precision@5 por query e por tamanho de corpus N, a partir dos
julgamentos de relevância preenchidos manualmente em data/pool_judgments.csv.

Pré-requisito:
    1. Rodar quality_analysis.py para gerar data/pool_judgments.csv.
    2. Preencher a coluna `relevant` com 1 (relevante) ou 0 (não relevante)
       para TODOS os documentos listados no arquivo.

Execução:
    python experiments/compute_precision.py

Saídas:
  results/raw/precision_at_5.csv
      Precision@5 por (query_id, N, config).

  results/raw/precision_summary.csv
      Precision@5 média por config e N (macro-average sobre as 5 queries).

Nota metodológica:
    Como C1, C2 e C3 usam a mesma função BM25 com os mesmos parâmetros e o
    mesmo corpus amostrado com a mesma semente, eles retornam ranking idêntico
    para qualquer (query, N). Logo, a coluna `top5_configs` do pool vai
    mostrar que qualquer documento do top-5 de C3 também pertence ao top-5 de
    C1 e C2 — e o Precision@5 das três configurações será numericamente igual.
    Isso é esperado e documenta que o BM25 com esses parâmetros é determinístico:
    o que as diferencia é tempo e memória, não a qualidade do ranking.
    C4 (rank_bm25) pode produzir ranking ligeiramente diferente por usar a
    fórmula completa de BM25 com o fator (k1+1); se produzir um top-5 distinto
    de C1/C2/C3, seu Precision@5 pode diferir.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "raw"
DATA_DIR = ROOT / "data"

POOL_PATH = DATA_DIR / "pool_judgments.csv"
PRECISION_PATH = RESULTS_DIR / "precision_at_5.csv"
SUMMARY_PATH = RESULTS_DIR / "precision_summary.csv"

CONFIGS = ("c1", "c2", "c3", "c4")
K = 5


def load_judgments() -> list[dict]:
    if not POOL_PATH.exists():
        print(
            f"Arquivo {POOL_PATH} não encontrado.\n"
            "Execute primeiro: python experiments/quality_analysis.py"
        )
        sys.exit(1)

    rows = []
    with open(POOL_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Verifica se todos os campos 'relevant' foram preenchidos
    unfilled = [
        r for r in rows if str(r.get("relevant", "")).strip() not in ("0", "1")
    ]
    if unfilled:
        print(
            f"Atenção: {len(unfilled)} entradas em pool_judgments.csv ainda não "
            "têm julgamento (relevant ≠ 0 e ≠ 1)."
        )
        print("Preencha a coluna 'relevant' antes de rodar este script.")
        print("Primeiras entradas sem julgamento:")
        for r in unfilled[:5]:
            print(f"  query_id={r['query_id']} N={r['N']} doc_id={r['doc_id']} "
                  f"title={str(r.get('title',''))[:60]}")
        sys.exit(1)

    return rows


def build_relevance_map(
    judgments: list[dict],
) -> dict[tuple[int, int, int], int]:
    """Retorna {(query_id, N, doc_id): relevant(0|1)}."""
    return {
        (int(r["query_id"]), int(r["N"]), int(r["doc_id"])): int(r["relevant"])
        for r in judgments
    }


def load_run_results() -> list[dict]:
    """Carrega os resultados brutos de results/raw/runs.csv para identificar
    quais doc_ids cada config retornou no top-k por (query_id, N, repetição)."""
    runs_path = RESULTS_DIR / "runs.csv"
    if not runs_path.exists():
        print(f"Arquivo {runs_path} não encontrado.")
        sys.exit(1)
    with open(runs_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_top5_from_pool(
    judgments: list[dict],
) -> dict[tuple[int, int, str], list[int]]:
    """Reconstrói o top-5 de cada config a partir do campo `top5_configs` do
    pool, sem precisar rerodar a busca.

    Retorna {(query_id, N, config): [doc_id, ...]} em ordem indefinida —
    suficiente para calcular Precision@K (que não depende da ordem dentro do
    top-k recuperado, só de quais documentos estão dentro).
    """
    # Mapeia (query_id, N) -> lista de doc_ids associados a cada config
    result: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    for r in judgments:
        q_id = int(r["query_id"])
        n = int(r["N"])
        doc_id = int(r["doc_id"])
        for cfg in r["top5_configs"].split("|"):
            cfg = cfg.strip()
            if cfg in CONFIGS:
                result[(q_id, n, cfg)].append(doc_id)
    return dict(result)


def precision_at_k(
    retrieved_doc_ids: list[int],
    relevance_map: dict[tuple[int, int, int], int],
    query_id: int,
    n: int,
    k: int = K,
) -> float:
    """P@k = (documentos relevantes dentro dos top-k recuperados) / k."""
    top_k = retrieved_doc_ids[:k]
    n_relevant = sum(
        relevance_map.get((query_id, n, doc_id), 0) for doc_id in top_k
    )
    return n_relevant / k if k > 0 else 0.0


def main() -> None:
    print("Carregando julgamentos de relevância...")
    judgments = load_judgments()
    relevance_map = build_relevance_map(judgments)
    top5_by_key = load_top5_from_pool(judgments)

    # Identifica combinações únicas de (N, query_id)
    n_values = sorted({int(r["N"]) for r in judgments})
    q_ids = sorted({int(r["query_id"]) for r in judgments})

    precision_rows: list[dict] = []

    for n in n_values:
        for q_id in q_ids:
            for cfg in CONFIGS:
                key = (q_id, n, cfg)
                retrieved = top5_by_key.get(key, [])
                if not retrieved:
                    # Config não retornou resultados para esse (query, N)
                    p5 = 0.0
                    n_relevant = 0
                else:
                    p5 = precision_at_k(retrieved, relevance_map, q_id, n)
                    n_relevant = sum(
                        relevance_map.get((q_id, n, d), 0) for d in retrieved[:K]
                    )
                precision_rows.append(
                    {
                        "query_id": q_id,
                        "N": n,
                        "config": cfg,
                        "n_retrieved": len(retrieved),
                        "n_relevant_in_top5": n_relevant,
                        "precision_at_5": round(p5, 4),
                    }
                )

    # Grava Precision@5 por query
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(precision_rows[0].keys())
    with open(PRECISION_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(precision_rows)
    print(f"Precision@5 por query salvo em: {PRECISION_PATH}")

    # Resumo: média sobre queries por (config, N)
    from collections import defaultdict
    sums: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in precision_rows:
        sums[(row["config"], row["N"])].append(float(row["precision_at_5"]))

    summary_rows: list[dict] = []
    for (cfg, n), values in sorted(sums.items()):
        avg = sum(values) / len(values)
        summary_rows.append(
            {
                "config": cfg,
                "N": n,
                "n_queries": len(values),
                "mean_precision_at_5": round(avg, 4),
            }
        )
    fieldnames_s = list(summary_rows[0].keys())
    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_s)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Resumo (média) salvo em: {SUMMARY_PATH}")

    # Imprime tabela de resumo no terminal
    print("\n── Precision@5 (média sobre 5 queries) ──────────────────────")
    print(f"{'config':<8} {'N':>8}  {'P@5 médio':>10}")
    print("-" * 32)
    for row in summary_rows:
        print(
            f"{row['config']:<8} {row['N']:>8}  {row['mean_precision_at_5']:>10.4f}"
        )


if __name__ == "__main__":
    main()
