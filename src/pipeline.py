"""Pipeline unificado das quatro configuracoes experimentais:

  C1 - baseline:               busca linear + Insertion Sort
  C2 - indexada:                índice invertido com hashmap + Insertion Sort
  C3 - divisão e conquista:    índice invertido com hashmap + Merge Sort
  C4 - baseline externo:        rank_bm25 (BM25Okapi) + sorted() do Python

Este modulo apenas monta e executa as configuracoes sobre um corpus ja
carregado. Nao mede tempo (time.perf_counter), nem memoria (tracemalloc),
nem calcula a precisao dos resultados.

Uso como CLI:
    python pipeline.py --config c2 --n 5000 --k 5 --query "impeachment de Dilma Rousseff"

Uso como biblioteca:
    from pipeline import (
        build_corpus,
        build_corpus_index,
        run,
    )
    from ingest import ensure_csv
    from normalize import clean_text, tokenize

    csv_path = ensure_csv("articles.csv")  # antes de iniciar a medição
    config = "c4"
    corpus = build_corpus(
        csv_path, sample_size=5000, seed=42, config=config
    )
    if config in ("c2", "c3"):
        build_corpus_index(corpus)

    query = tokenize(clean_text("impeachment de Dilma Rousseff"))
    result = run(config, corpus, query, k=5)

    print(result.results)
    print(result.search_comparisons)
    print(result.sort_comparisons)
    print(result.candidate_count)
"""

import argparse
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from bm25 import BM25Stats, build_bm25_stats
from ingest import ensure_csv, load_and_process_corpus
from inverted_index import InvertedIndex, build_inverted_index, indexed_search
from linear_search import linear_search
from normalize import clean_text, tokenize
from sorting import insertion_sort, merge_sort

if TYPE_CHECKING:
    import pandas as pd

CONFIGS: tuple[str, ...] = ("c1", "c2", "c3", "c4")


@dataclass
class Corpus:
    """Estado construido a partir do corpus (ingestao e estatisticas do
    BM25) e reaproveitado por todas as consultas e configuracoes. O
    indice invertido e opcional e so precisa ser construido quando as
    configuracoes C2 ou C3 forem usadas."""

    df: "pd.DataFrame"
    documents_tokens: list[list[str]]
    bm25_stats: BM25Stats
    inverted_index: InvertedIndex | None = None
    _bm25_reference: BM25Okapi | None = field(default=None, repr=False)


@dataclass
class ExecutionResult:
    """Saida padronizada de qualquer configuracao (C1 a C4), pronta para
    ser consumida por quem for medir tempo, memoria ou calcular a
    precisao dos resultados."""

    config: str
    results: list[tuple[int, float]]  # (doc_id, score), top-k, ja ordenado
    search_comparisons: int
    sort_comparisons: int
    candidate_count: int  # documentos com score > 0, antes do corte por k


def _require_index(corpus: Corpus) -> InvertedIndex:
    if corpus.inverted_index is None:
        raise ValueError(
            "corpus.inverted_index e None -- chame build_corpus_index(corpus) "
            "antes de executar as Configuracoes C2/C3."
        )
    return corpus.inverted_index


def _get_bm25_reference(corpus: Corpus) -> BM25Okapi:
    """Retorna a instância do BM25Okapi preparada para a configuração
    C4 durante a construção do corpus."""
    if corpus._bm25_reference is None:
        raise ValueError(
            "A referência BM25Okapi não foi preparada. "
            "Use build_corpus(..., config='c4') antes de executar C4."
        )
    return corpus._bm25_reference


def _postprocess(
    sorted_results: list[tuple[int, float]], k: int
) -> tuple[list[tuple[int, float]], int]:
    """Mantem apenas os candidatos com score maior que zero e corta a
    lista em k itens. Tambem devolve quantos candidatos positivos
    existiam antes desse corte."""
    positive = [pair for pair in sorted_results if pair[1] > 0]
    return positive[:k], len(positive)


def build_corpus(
    csv_path: str = "articles.csv",
    sample_size: int | None = 5000,
    seed: int = 42,
    config: str = "c1",
) -> Corpus:
    """Realiza a ingestão do corpus: leitura do CSV, amostragem,
    limpeza, tokenização e cálculo das estatísticas do BM25.

    Quando config é C4, também constrói o BM25Okapi durante esta etapa,
    para que a primeira chamada a run("c4", ...) não tenha custo de
    inicialização. Para C1, C2 e C3, essa construção não é realizada.
    O índice invertido continua sendo construído separadamente por
    build_corpus_index(), quando necessário para C2 e C3."""
    if config not in CONFIGS:
        raise ValueError(
            f"Configuração desconhecida: {config!r}. Use uma de {CONFIGS}."
        )

    df = load_and_process_corpus(csv_path, sample_size=sample_size, seed=seed)
    documents_tokens = df["tokens"].tolist()
    bm25_stats = build_bm25_stats(documents_tokens)
    corpus = Corpus(df, documents_tokens, bm25_stats)

    if config == "c4":
        corpus._bm25_reference = BM25Okapi(documents_tokens)

    return corpus


def build_corpus_index(corpus: Corpus) -> None:
    """Constrói o índice invertido com hashmap a partir das estatísticas
    já calculadas por build_corpus() e o armazena no próprio objeto
    Corpus. Necessário apenas para as configurações C2 e C3; C1 e C4
    não utilizam o índice invertido."""
    corpus.inverted_index = build_inverted_index(corpus.bm25_stats)


def run_c1(corpus: Corpus, query_tokens: list[str], k: int) -> ExecutionResult:
    """Executa a configuracao C1 (baseline): busca linear seguida de
    Insertion Sort."""
    candidates, search_comparisons = linear_search(query_tokens, corpus.bm25_stats)
    sorted_candidates, sort_comparisons = insertion_sort(candidates)
    top_k, candidate_count = _postprocess(sorted_candidates, k)
    return ExecutionResult(
        "c1", top_k, search_comparisons, sort_comparisons, candidate_count
    )


def run_c2(corpus: Corpus, query_tokens: list[str], k: int) -> ExecutionResult:
    """Executa a configuração C2 (indexada): consulta um índice
    invertido com hashmap e aplica Insertion Sort aos candidatos
    encontrados."""
    candidates, search_comparisons = indexed_search(
        query_tokens, _require_index(corpus), corpus.bm25_stats
    )
    sorted_candidates, sort_comparisons = insertion_sort(candidates)
    top_k, candidate_count = _postprocess(sorted_candidates, k)
    return ExecutionResult(
        "c2", top_k, search_comparisons, sort_comparisons, candidate_count
    )


def run_c3(corpus: Corpus, query_tokens: list[str], k: int) -> ExecutionResult:
    """Executa a configuração C3 (divisão e conquista): consulta o
    mesmo índice invertido da configuração C2, mas usa Merge Sort no
    lugar do Insertion Sort."""
    candidates, search_comparisons = indexed_search(
        query_tokens, _require_index(corpus), corpus.bm25_stats
    )
    sorted_candidates, sort_comparisons = merge_sort(candidates)
    top_k, candidate_count = _postprocess(sorted_candidates, k)
    return ExecutionResult(
        "c3", top_k, search_comparisons, sort_comparisons, candidate_count
    )


def run_c4(corpus: Corpus, query_tokens: list[str], k: int) -> ExecutionResult:
    """Executa a configuracao C4 (baseline externo): usa a biblioteca
    rank_bm25 (BM25Okapi) para calcular os scores e a funcao sorted()
    nativa do Python para ordena-los."""
    bm25_reference = _get_bm25_reference(corpus)
    scores = bm25_reference.get_scores(query_tokens)
    candidates = [(doc_id, float(score)) for doc_id, score in enumerate(scores)]
    sorted_candidates = sorted(candidates, key=lambda pair: (-pair[1], pair[0]))
    top_k, candidate_count = _postprocess(sorted_candidates, k)
    return ExecutionResult("c4", top_k, 0, 0, candidate_count)


_RUNNERS = {
    "c1": run_c1,
    "c2": run_c2,
    "c3": run_c3,
    "c4": run_c4,
}


def run(
    config: str, corpus: Corpus, query_tokens: list[str], k: int
) -> ExecutionResult:
    """Ponto de entrada unico para executar qualquer configuracao:
    recebe o identificador (c1, c2, c3 ou c4) e despacha a chamada para
    a funcao correspondente, mantendo a mesma assinatura para todas."""
    try:
        runner = _RUNNERS[config]
    except KeyError:
        raise ValueError(
            f"Configuracao desconhecida: {config!r}. Use uma de {CONFIGS}."
        ) from None
    return runner(corpus, query_tokens, k)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa uma configuracao (C1 a C4) de busca e ordenacao BM25 sobre o corpus."
    )
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--csv", default="articles.csv")
    parser.add_argument(
        "--n", type=int, default=5000, help="tamanho da amostra do corpus (N)"
    )
    parser.add_argument(
        "--k", type=int, default=5, help="numero de resultados retornados"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    csv_path = ensure_csv(args.csv)
    corpus = build_corpus(
        csv_path,
        sample_size=args.n,
        seed=args.seed,
        config=args.config,
    )

    if args.config in ("c2", "c3"):
        build_corpus_index(corpus)
    query_tokens = tokenize(clean_text(args.query))

    result = run(args.config, corpus, query_tokens, args.k)

    print(f"\nConfiguracao: {result.config.upper()}")
    print(f"Consulta: '{args.query}' -> tokens normalizados: {query_tokens}")
    print(f"Candidatos com score > 0: {result.candidate_count}")
    print(f"Comparacoes (busca): {result.search_comparisons}")
    print(f"Comparacoes (ordenacao): {result.sort_comparisons}\n")
    print(f"Top-{args.k}:")
    for doc_id, score in result.results:
        row = corpus.df.iloc[doc_id]
        print(f"  [score {score:.3f}] {row['title']}  ({row['date']})")


if __name__ == "__main__":
    main()
