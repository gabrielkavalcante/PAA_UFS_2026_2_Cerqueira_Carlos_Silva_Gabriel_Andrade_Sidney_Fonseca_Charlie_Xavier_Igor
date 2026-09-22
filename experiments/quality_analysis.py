"""Análise de qualidade: estatísticas de tokens, vocabulary mismatch e geração
do pool de julgamentos para o Precision@5.

Execução:
    python experiments/quality_analysis.py

Saídas produzidas:
  data/pool_judgments.csv
      Pool de documentos únicos do top-5 de C1+C2+C3+C4 para cada
      combinação (query_id, N). A coluna `relevant` fica VAZIA — o avaliador
      deve preenchê-la manualmente com 1 (relevante) ou 0 (não relevante)
      antes de rodar compute_precision.py.

  results/raw/token_stats.csv
      Estatísticas de comprimento dos documentos por tamanho de corpus.

  results/raw/vocab_mismatch.csv
      Exemplos de vocabulary mismatch detectados automaticamente para cada
      query: termos normalizados da query que aparecem em ZERO documentos do
      top-5 retornado por C3. Indica termos "perdidos" — a query menciona uma
      palavra que o BM25 não conseguiu usar para recuperar nenhum documento.
"""

import csv
import sys
import statistics
from pathlib import Path

# Resolve o caminho do src e o adiciona ao PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from ingest import ensure_csv, load_and_process_corpus
from normalize import clean_text, tokenize
from pipeline import build_corpus, build_corpus_index, run

# ── Parâmetros ────────────────────────────────────────────────────────────────
# Usa os mesmos tamanhos que run_experiments.py para manter coerência.
N_VALUES = [1000, 5000, 10000, 25000, 50000]
K = 5
SEED = 42

QUERIES: list[tuple[int, str]] = [
    (1, "impeachment de Dilma Rousseff"),
    (2, "operação Lava Jato delação"),
    (3, "reforma da previdência"),
    (4, "crise hídrica em São Paulo"),
    (5, "eleição municipal 2016"),
]

# Configs incluídas no pool de julgamento
POOL_CONFIGS = ("c1", "c2", "c3", "c4")

# ── Diretórios de saída ───────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_token_stats(documents_tokens: list[list[str]]) -> dict[str, float]:
    lengths = [len(tokens) for tokens in documents_tokens]
    if not lengths:
        return {}
    return {
        "n_docs": len(lengths),
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "stdev": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
        "min": min(lengths),
        "max": max(lengths),
    }


def detect_vocab_mismatch(
    query_tokens: list[str],
    top5_doc_ids: list[int],
    corpus_tokens: list[list[str]],
) -> list[str]:
    """Retorna os termos da query que não aparecem em NENHUM dos top-5
    documentos recuperados. Esses são os termos 'perdidos' pelo BM25 —
    a query os usa, mas os documentos retornados não os contêm."""
    missing: list[str] = []
    for term in set(query_tokens):
        found_in_any = any(
            term in set(corpus_tokens[doc_id]) for doc_id in top5_doc_ids
        )
        if not found_in_any:
            missing.append(term)
    return sorted(missing)


def main() -> None:
    print("Verificando dataset...")
    csv_path = ensure_csv()

    # ── Arquivo de estatísticas de tokens ────────────────────────────────────
    token_stats_path = RESULTS_DIR / "token_stats.csv"
    token_stats_rows: list[dict] = []

    # ── Pool de julgamentos ───────────────────────────────────────────────────
    pool_path = DATA_DIR / "pool_judgments.csv"
    # pool_rows[(query_id, N, doc_id)] = {metadados} — deduplicação por doc_id
    # dentro do mesmo (query_id, N)
    pool_by_key: dict[tuple[int, int, int], dict] = {}

    # ── Vocabulary mismatch ───────────────────────────────────────────────────
    mismatch_path = RESULTS_DIR / "vocab_mismatch.csv"
    mismatch_rows: list[dict] = []

    for n in N_VALUES:
        print(f"\n=== N = {n} ===")

        # Estatísticas de tokens (independente de config)
        print(f"  Calculando estatísticas de tokens...")
        corpus_c3 = build_corpus(csv_path, sample_size=n, seed=SEED, config="c3")
        build_corpus_index(corpus_c3)
        stats = compute_token_stats(corpus_c3.documents_tokens)
        token_stats_rows.append({"N_solicitado": n, **stats})
        print(
            f"  tokens/doc: média={stats['mean']:.1f}, "
            f"mediana={stats['median']:.0f}, "
            f"max={stats['max']}"
        )

        # Corpora adicionais para as outras configs (mesmo seed, mesmos docs)
        corpus_c1 = build_corpus(csv_path, sample_size=n, seed=SEED, config="c1")
        corpus_c4 = build_corpus(csv_path, sample_size=n, seed=SEED, config="c4")
        # C2 reutiliza o mesmo índice que C3
        corpus_c2 = corpus_c3

        corpora = {"c1": corpus_c1, "c2": corpus_c2, "c3": corpus_c3, "c4": corpus_c4}

        for q_id, q_text in QUERIES:
            query_tokens = tokenize(clean_text(q_text))
            print(f"  Query {q_id}: '{q_text}' -> tokens: {query_tokens}")

            # Coleta top-5 de cada config e registra quais configs retornaram cada doc
            doc_to_configs: dict[int, list[str]] = {}
            all_top5_ids: list[int] = []

            for cfg in POOL_CONFIGS:
                result = run(cfg, corpora[cfg], query_tokens, K)
                for doc_id, score in result.results:
                    doc_to_configs.setdefault(doc_id, []).append(cfg)
                    if doc_id not in all_top5_ids:
                        all_top5_ids.append(doc_id)

            # Vocabulary mismatch: usa C3 (índice invertido + Merge Sort)
            # como referência de recuperação
            result_c3 = run("c3", corpus_c3, query_tokens, K)
            top5_c3_ids = [doc_id for doc_id, _ in result_c3.results]
            missing_terms = detect_vocab_mismatch(
                query_tokens, top5_c3_ids, corpus_c3.documents_tokens
            )
            if missing_terms:
                print(f"    Mismatch (termos perdidos no top-5 C3): {missing_terms}")
            mismatch_rows.append(
                {
                    "N": n,
                    "query_id": q_id,
                    "query_text": q_text,
                    "query_tokens": " ".join(query_tokens),
                    "n_query_tokens": len(query_tokens),
                    "missing_terms": " ".join(missing_terms) if missing_terms else "",
                    "n_missing": len(missing_terms),
                    "top5_c3_doc_ids": " ".join(str(d) for d in top5_c3_ids),
                }
            )

            # Pool de julgamentos
            for doc_id in all_top5_ids:
                key = (q_id, n, doc_id)
                if key not in pool_by_key:
                    row_df = corpus_c1.df.iloc[doc_id]
                    pool_by_key[key] = {
                        "query_id": q_id,
                        "query_text": q_text,
                        "N": n,
                        "doc_id": doc_id,
                        "title": row_df.get("title", ""),
                        "date": row_df.get("date", ""),
                        "link": row_df.get("link", ""),
                        "top5_configs": "|".join(sorted(doc_to_configs.get(doc_id, []))),
                        "relevant": "",  # <── preencher manualmente (1 ou 0)
                    }

    # ── Grava arquivos de saída ───────────────────────────────────────────────

    # Token stats
    if token_stats_rows:
        fieldnames = list(token_stats_rows[0].keys())
        with open(token_stats_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(token_stats_rows)
        print(f"\nEstatísticas de tokens salvas em: {token_stats_path}")

    # Pool de julgamentos
    if pool_by_key:
        pool_rows = list(pool_by_key.values())
        fieldnames = list(pool_rows[0].keys())
        with open(pool_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pool_rows)
        print(f"Pool de julgamentos salvo em: {pool_path}")
        print(
            f"  → {len(pool_rows)} pares (query, N, doc) aguardando julgamento manual "
            f"na coluna 'relevant'."
        )

    # Vocabulary mismatch
    if mismatch_rows:
        fieldnames = list(mismatch_rows[0].keys())
        with open(mismatch_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mismatch_rows)
        print(f"Vocabulary mismatch salvo em: {mismatch_path}")

    print("\nConcluído. Preencha a coluna 'relevant' em data/pool_judgments.csv")
    print("e então execute: python experiments/compute_precision.py")


if __name__ == "__main__":
    main()
