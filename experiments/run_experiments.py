import os
import sys
import time
import csv
import tracemalloc
from pathlib import Path

caminho_src = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, caminho_src)

from ingest import ensure_csv
from normalize import clean_text, tokenize
from pipeline import build_corpus, build_corpus_index, run

N_VALUES = [5000, 50000]
CONFIGS = ["c1", "c2", "c3", "c4"]
K = 5

REPETITIONS = 3


QUERIES = [
    "impeachment de Dilma Rousseff",
    "operação Lava Jato delação",
    "reforma da previdência",
    "crise hídrica em São Paulo",
    "eleição municipal 2016"
]

def main():
    
    pasta_resultados = Path(__file__).resolve().parent.parent / "results" / "raw"
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    
    arquivo_csv = pasta_resultados / "runs.csv"

    
    print("Verificando dataset...")
    csv_path = ensure_csv()

   
    cabecalho = [
        "config", "N", "k", "query_id", "repeticao", 
        "t_ingestao_ms", "t_indice_ms", "t_consulta_ms", "t_total_ms", 
        "memoria_pico_mb", "n_comparacoes", "n_resultados", "resultado_vazio"
    ]

    
    with open(arquivo_csv, mode="w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(cabecalho) 
        for n in N_VALUES:
            print(f"\n--- Iniciando testes para N = {n} ---")
            
          
            t0 = time.perf_counter()
            corpus_base = build_corpus(csv_path, sample_size=n, seed=42, config="c1")
            t_ingestao_ms = (time.perf_counter() - t0) * 1000

           
            corpus_indexado = build_corpus(csv_path, sample_size=n, seed=42, config="c2")
            t0 = time.perf_counter()
            build_corpus_index(corpus_indexado)
            t_indice_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            corpus_c4 = build_corpus(csv_path, sample_size=n, seed=42, config="c4")
            t_total_c4_setup = (time.perf_counter() - t0) * 1000
            
            
            t_setup_rankbm25_ms = max(0.0, t_total_c4_setup - t_ingestao_ms)

            for config in CONFIGS:
                print(f"  Rodando configuração: {config.upper()}")
                
                if config in ["c2", "c3"]:
                    corpus_atual = corpus_indexado
                    tempo_indice_atual = t_indice_ms
                elif config == "c4":
                    corpus_atual = corpus_c4
                    tempo_indice_atual = t_setup_rankbm25_ms
                else: 
                    corpus_atual = corpus_base
                    tempo_indice_atual = 0.0

                for q_idx, texto_query in enumerate(QUERIES, 1):
            
                    query_tokens = tokenize(clean_text(texto_query))

                    _ = run(config, corpus_atual, query_tokens, K)


                    for rep in range(1, REPETITIONS + 1):
                        
                        tracemalloc.start() 
                        inicio_cronometro = time.perf_counter()
                        
                        resultado = run(config, corpus_atual, query_tokens, K)
                        
                        t_consulta_ms = (time.perf_counter() - inicio_cronometro) * 1000

                        _, memoria_pico_bytes = tracemalloc.get_traced_memory()
                        tracemalloc.stop()

                        mem_mb = memoria_pico_bytes / (1024 * 1024)
                        t_total_ms = t_ingestao_ms + tempo_indice_atual + t_consulta_ms
                        
                        n_comparacoes = resultado.search_comparisons + resultado.sort_comparisons
                        

                        vazio = 1 if len(resultado.results) == 0 else 0

                        escritor.writerow([
                            config, n, K, q_idx, rep,
                            round(t_ingestao_ms, 3), 
                            round(tempo_indice_atual, 3), 
                            round(t_consulta_ms, 3), 
                            round(t_total_ms, 3),
                            round(mem_mb, 4), 
                            n_comparacoes, 
                            len(resultado.results), 
                            vazio
                        ])
                        
    print(f"\nExperimentos concluídos! Vá na pasta 'results/raw' e abra o 'runs.csv'.")

if __name__ == "__main__":
    main()    