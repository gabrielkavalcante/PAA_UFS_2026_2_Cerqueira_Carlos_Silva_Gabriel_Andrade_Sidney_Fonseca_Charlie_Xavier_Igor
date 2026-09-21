import csv
import sys
import time
import tracemalloc
from pathlib import Path

caminho_src = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, caminho_src)

from ingest import ensure_csv
from normalize import clean_text, tokenize
from pipeline import build_corpus, build_corpus_index, run

N_VALUES = [1000, 5000, 10000, 25000, 50000]
CONFIGS = ["c1", "c2", "c3", "c4"]
K = 5
REPETITIONS = 3

QUERIES = [
    "impeachment de Dilma Rousseff",
    "operação Lava Jato delação",
    "reforma da previdência",
    "crise hídrica em São Paulo",
    "eleição municipal 2016",
]


def medir_build(csv_path: str, n: int, config: str):
    """Constrói um corpus e devolve o corpus e o tempo de preparação."""
    inicio = time.perf_counter()
    corpus = build_corpus(csv_path, sample_size=n, seed=42, config=config)
    tempo_ms = (time.perf_counter() - inicio) * 1000
    return corpus, tempo_ms


def main():
    pasta_resultados = Path(__file__).resolve().parent.parent / "results" / "raw"
    pasta_resultados.mkdir(parents=True, exist_ok=True)

    arquivo_csv = pasta_resultados / "runs.csv"

    print("Verificando dataset...")
    csv_path = ensure_csv(str(Path(__file__).resolve().parent.parent / "articles.csv"))

    cabecalho = [
        "config", "N", "n_documentos_reais", "k", "query_id", "repeticao",
        "t_ingestao_ms", "t_indice_ms", "t_consulta_ms",
        "t_total_ms", "memoria_pico_consulta_mb", "n_comparacoes",
        "n_resultados", "resultado_vazio"
    ]  # fmt: skip

    with open(arquivo_csv, mode="w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(cabecalho)
        tamanhos_testados = set()

        for n in N_VALUES:
            print(f"\n--- Iniciando testes para N = {n} ---")

            # Cada configuração tem seu próprio custo de ingestão medido.
            corpus_base, t_ingestao_c1_ms = medir_build(csv_path, n, "c1")
            corpus_indexado, t_ingestao_c2_ms = medir_build(csv_path, n, "c2")

            inicio = time.perf_counter()
            build_corpus_index(corpus_indexado)
            t_indice_c2_ms = (time.perf_counter() - inicio) * 1000

            # Para C4, build_corpus inclui a construção do BM25Okapi.
            corpus_c4, t_bm25_c4_ms = medir_build(csv_path, n, "c4")

            tamanhos = {
                len(corpus_base.documents_tokens),
                len(corpus_indexado.documents_tokens),
                len(corpus_c4.documents_tokens),
            }
            if len(tamanhos) != 1:
                raise RuntimeError(
                    "Os corpora das configurações ficaram com tamanhos diferentes."
                )
            n_documentos_reais = tamanhos.pop()
            if n_documentos_reais in tamanhos_testados:
                print(
                    f"  Ignorando N = {n}: "
                    f"corpus real repetido ({n_documentos_reais} documentos)."
                )
                continue
            tamanhos_testados.add(n_documentos_reais)

            tempos_setup = {
                "c1": (t_ingestao_c1_ms, 0.0),
                "c2": (t_ingestao_c2_ms, t_indice_c2_ms),
                "c3": (t_ingestao_c2_ms, t_indice_c2_ms),
                # Em C4, o tempo de ingestão inclui a construção do BM25Okapi.
                "c4": (t_bm25_c4_ms, 0.0),
            }
            corpora = {
                "c1": corpus_base,
                "c2": corpus_indexado,
                "c3": corpus_indexado,
                "c4": corpus_c4,
            }

            for config in CONFIGS:
                print(f"  Rodando configuração: {config.upper()}")
                corpus_atual = corpora[config]
                t_ingestao_ms, t_indice_ms = tempos_setup[config]

                for q_idx, texto_query in enumerate(QUERIES, 1):
                    query_tokens = tokenize(clean_text(texto_query))

                    # Aquece a execução sem incluí-la nas medições.
                    run(config, corpus_atual, query_tokens, K)

                    for rep in range(1, REPETITIONS + 1):
                        # Mede o tempo sem o overhead do tracemalloc.
                        inicio_cronometro = time.perf_counter()
                        resultado = run(config, corpus_atual, query_tokens, K)
                        t_consulta_ms = (time.perf_counter() - inicio_cronometro) * 1000

                        # Mede memória em uma execução separada, para não alterar
                        # o tempo registrado acima. O valor é apenas da consulta.
                        tracemalloc.start()
                        run(config, corpus_atual, query_tokens, K)
                        _, memoria_pico_bytes = tracemalloc.get_traced_memory()
                        tracemalloc.stop()

                        mem_mb = memoria_pico_bytes / (1024 * 1024)
                        t_total_ms = t_ingestao_ms + t_indice_ms + t_consulta_ms

                        n_comparacoes = (
                            resultado.search_comparisons + resultado.sort_comparisons
                        )
                        vazio = 1 if len(resultado.results) == 0 else 0

                        escritor.writerow(
                            [
                                config,
                                n,
                                n_documentos_reais,
                                K,
                                q_idx,
                                rep,
                                round(t_ingestao_ms, 3),
                                round(t_indice_ms, 3),
                                round(t_consulta_ms, 3),
                                round(t_total_ms, 3),
                                round(mem_mb, 4),
                                n_comparacoes,
                                len(resultado.results),
                                vazio,
                            ]
                        )

    print("\nExperimentos concluídos! Vá na pasta 'results/raw' e abra o 'runs.csv'.")


if __name__ == "__main__":
    main()
