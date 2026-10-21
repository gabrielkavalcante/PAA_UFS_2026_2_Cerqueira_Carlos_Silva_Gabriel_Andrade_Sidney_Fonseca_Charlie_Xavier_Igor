"""AV2 (Trilha A, mochila 0/1): roda a grade completa de experimentos.

Para cada N, consulta, W e método mede o tempo de seleção (3 repetições
após 1 aquecimento descartado) e a memória de pico, calcula as métricas
de qualidade do contexto escolhido e grava:

    results/raw/av2_runs.csv        450 linhas na grade completa
    results/raw/av2_validacao.csv   30 instâncias sintéticas (DP x força bruta)
    results/raw/av2_ambiente.txt    ambiente, parâmetros e comando executado
    results/raw/av2_faltando.txt    só existe se algum módulo não foi importado

Uso (a partir da raiz do repositório):
    uv run python experiments/run_av2.py
"""

import argparse
import csv
import importlib
import os
import platform
import random
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime
from importlib import metadata
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from av2_types import Candidate, Selection  # noqa: E402
from ingest import ensure_csv  # noqa: E402
from normalize import clean_text, tokenize  # noqa: E402
from pipeline import build_corpus, build_corpus_index  # noqa: E402

# Parâmetros fixos da AV2 (aba "Decisões fixas" do plano).
N_VALUES = [10_000, 50_000]
W_VALUES = [500, 1_000, 2_000]
REPETITIONS = 3
M_CANDIDATES = 200
SEED = 42
VALIDATION_INSTANCES = 30
VALIDATION_ITEMS = 15

# Colunas fixas de results/raw/av2_runs.csv (aba "Código fixo").
COLUMNS = [
    "method", "N", "n_docs_reais", "query_id", "W", "rep", "n_candidatos",
    "t_preproc_ms", "t_indice_ms", "t_candidatos_ms", "t_selecao_ms",
    "t_total_ms", "memoria_pico_mb", "ops", "n_docs_sel", "tokens_used",
    "budget_ok", "total_score", "coverage", "redundancy",
]  # fmt: skip

VALIDATION_COLUMNS = [
    "instance_id", "n_items", "W", "dp_value", "brute_value",
    "greedy_ratio_value", "greedy_ratio_plus_value", "greedy_score_value",
    "dp_igual_brute",
]  # fmt: skip

# método -> (módulo, função), na ordem em que entram no CSV.
METHOD_SPECS = {
    "topk_av1": ("topk_av1", "solve"),
    "greedy_score": ("knapsack_greedy", "solve_score"),
    "greedy_ratio": ("knapsack_greedy", "solve_ratio"),
    "greedy_ratio_plus": ("knapsack_greedy", "solve_ratio_plus"),
    "dp": ("knapsack_dp", "solve"),
}


def import_attr(module_name: str, attr: str, missing: list[str]):
    """Importa module_name.attr; se falhar, registra o nome em `missing`
    e devolve None."""
    try:
        return getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as erro:
        missing.append(f"{module_name}.{attr}  ({type(erro).__name__}: {erro})")
        return None


def synthetic_instance(
    seed: int,
    n_items: int,
    cost_range: tuple[int, int] = (1, 10),
    value_range: tuple[float, float] = (1.0, 10.0),
) -> tuple[int, list[Candidate]]:
    """Instância sintética reprodutível: devolve (W, candidatos), com
    W = metade da soma dos custos."""
    rng = random.Random(seed)
    candidates = [
        Candidate(
            doc_id,
            round(rng.uniform(*value_range), 1),
            rng.randint(*cost_range),
        )
        for doc_id in range(n_items)
    ]
    budget = sum(c.cost for c in candidates) // 2
    return budget, candidates


def measure_selection(solve, candidates: list[Candidate], budget: int):
    """Uma execução cronometrada. Devolve (Selection, tempo em ms)."""
    inicio = time.perf_counter()
    selection = solve(candidates, budget)
    return selection, (time.perf_counter() - inicio) * 1000


def measure_peak_memory_mb(solve, candidates: list[Candidate], budget: int) -> float:
    """Memória de pico de uma execução à parte, como na AV1."""
    tracemalloc.start()
    solve(candidates, budget)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def load_queries(path: Path, wanted: list[int] | None) -> list[tuple[int, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [(int(r["query_id"]), r["query_text"]) for r in csv.DictReader(f)]
    if wanted:
        rows = [row for row in rows if row[0] in wanted]
    return rows


def run_grid(args, out_dir: Path, methods: dict, build_candidates, evaluate) -> int:
    """Roda a grade de experimentos e grava av2_runs.csv. Devolve o
    número de linhas gravadas."""
    csv_path = ensure_csv(args.csv)
    queries = load_queries(RAIZ / "data" / "queries.csv", args.queries)
    n_rows = 0
    real_sizes_seen: set[int] = set()

    with open(out_dir / "av2_runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)

        for n in args.n:
            print(f"\n--- N = {n} ---")
            inicio = time.perf_counter()
            corpus = build_corpus(csv_path, sample_size=n, seed=SEED, config="c2")
            t_preproc_ms = (time.perf_counter() - inicio) * 1000
            inicio = time.perf_counter()
            build_corpus_index(corpus)
            t_indice_ms = (time.perf_counter() - inicio) * 1000

            n_reais = len(corpus.documents_tokens)
            if n_reais in real_sizes_seen:
                print(f"  Ignorando N = {n}: corpus real repetido ({n_reais} docs).")
                continue
            real_sizes_seen.add(n_reais)

            for query_id, query_text in queries:
                query_tokens = tokenize(clean_text(query_text))
                inicio = time.perf_counter()
                candidates = build_candidates(corpus, query_tokens, m=args.m)
                t_candidatos_ms = (time.perf_counter() - inicio) * 1000
                print(f"  q{query_id}: {len(candidates)} candidatos")

                for budget in args.w:
                    for name, solve in methods.items():
                        solve(candidates, budget)  # aquecimento descartado
                        for rep in range(1, args.reps + 1):
                            selection, t_selecao_ms = measure_selection(
                                solve, candidates, budget
                            )
                            memoria_mb = measure_peak_memory_mb(
                                solve, candidates, budget
                            )
                            m = evaluate(
                                selection, budget, query_tokens,
                                corpus.documents_tokens,
                            )  # fmt: skip
                            t_total_ms = (
                                t_preproc_ms + t_indice_ms
                                + t_candidatos_ms + t_selecao_ms
                            )  # fmt: skip
                            writer.writerow([
                                name, n, n_reais, query_id, budget, rep,
                                len(candidates),
                                round(t_preproc_ms, 3), round(t_indice_ms, 3),
                                round(t_candidatos_ms, 3), round(t_selecao_ms, 3),
                                round(t_total_ms, 3), round(memoria_mb, 4),
                                selection.ops, m["n_docs_sel"], m["tokens_used"],
                                1 if m["budget_ok"] else 0,
                                round(m["total_score"], 6),
                                round(m["coverage"], 6), round(m["redundancy"], 6),
                            ])  # fmt: skip
                            n_rows += 1
                f.flush()
    return n_rows


def run_validation(out_dir: Path, dp, brute, ratio, ratio_plus, score) -> None:
    """30 instâncias sintéticas: compara o DP com a força bruta e com
    os gulosos. Colunas de métodos ausentes ficam vazias."""

    def valor(solve, budget, candidates):
        return "" if solve is None else round(solve(candidates, budget).total_value, 6)

    with open(out_dir / "av2_validacao.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(VALIDATION_COLUMNS)
        for seed in range(VALIDATION_INSTANCES):
            budget, candidates = synthetic_instance(seed, VALIDATION_ITEMS)
            dp_v = valor(dp, budget, candidates)
            brute_v = valor(brute, budget, candidates)
            igual = "" if "" in (dp_v, brute_v) else int(abs(dp_v - brute_v) <= 1e-9)
            writer.writerow([
                seed, VALIDATION_ITEMS, budget, dp_v, brute_v,
                valor(ratio, budget, candidates),
                valor(ratio_plus, budget, candidates),
                valor(score, budget, candidates), igual,
            ])  # fmt: skip


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for linha in f:
                if linha.startswith("model name"):
                    return linha.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "desconhecido"


def memory_total_gb() -> str:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            kb = int(f.readline().split()[1])
        return f"{kb / (1024 * 1024):.1f} GB"
    except (OSError, ValueError, IndexError):
        return "desconhecida"


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "não instalado"


def commit_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True,
            text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "desconhecido"


def write_environment(out_dir: Path, args, n_rows: int) -> None:
    linhas = [
        "AV2 - ambiente da rodada de experimentos",
        f"data: {datetime.now().isoformat(timespec='seconds')}",
        f"commit: {commit_hash()}",
        f"sistema operacional: {platform.platform()}",
        f"python: {platform.python_version()}",
        f"pandas: {package_version('pandas')}",
        f"nltk: {package_version('nltk')}",
        f"rank-bm25: {package_version('rank-bm25')}",
        f"matplotlib: {package_version('matplotlib')}",
        f"processador: {cpu_name()}",
        f"nucleos: {os.cpu_count()}",
        f"memoria: {memory_total_gb()}",
        f"parametros: N={args.n} W={args.w} M={args.m} reps={args.reps} "
        f"consultas={args.queries or 'todas'} seed={SEED}",
        f"linhas em av2_runs.csv: {n_rows}",
        f"comando: {' '.join(sys.argv)}",
    ]
    (out_dir / "av2_ambiente.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", default=str(RAIZ / "articles.csv"))
    parser.add_argument("--n", type=int, nargs="+", default=N_VALUES)
    parser.add_argument("--w", type=int, nargs="+", default=W_VALUES)
    parser.add_argument("--queries", type=int, nargs="+", default=None,
                        help="ids das consultas (padrão: todas, q1 a q5)")  # fmt: skip
    parser.add_argument("--reps", type=int, default=REPETITIONS)
    parser.add_argument("--m", type=int, default=M_CANDIDATES)
    parser.add_argument("--out-dir", default=str(RAIZ / "results" / "raw"))
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    methods = {}
    for name, (module_name, attr) in METHOD_SPECS.items():
        solve = import_attr(module_name, attr, missing)
        if solve is not None:
            methods[name] = solve
    build_candidates = import_attr("av2_candidates", "build_candidates", missing)
    evaluate = import_attr("metrics", "evaluate", missing)
    brute = import_attr("brute_force", "solve", missing)

    faltando = out_dir / "av2_faltando.txt"
    if missing:
        faltando.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"Aviso: {len(missing)} import(s) falharam; veja {faltando}")
    elif faltando.exists():
        faltando.unlink()

    if build_candidates is None or evaluate is None:
        print("Erro: sem av2_candidates.build_candidates e metrics.evaluate "
              "não há como montar a grade.")  # fmt: skip
        return 1

    n_rows = run_grid(args, out_dir, methods, build_candidates, evaluate)
    if not args.skip_validation:
        run_validation(
            out_dir, methods.get("dp"), brute,
            methods.get("greedy_ratio"), methods.get("greedy_ratio_plus"),
            methods.get("greedy_score"),
        )  # fmt: skip
    write_environment(out_dir, args, n_rows)
    print(f"\nConcluído: {n_rows} linhas em {out_dir / 'av2_runs.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
