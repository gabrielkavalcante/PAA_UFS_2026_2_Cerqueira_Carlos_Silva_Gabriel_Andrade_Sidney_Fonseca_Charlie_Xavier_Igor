from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def gerar_grafico(df, configs, coluna, titulo, ylabel, escala, caminho):
    plt.figure(figsize=(10, 6))
    for config in configs:
        dados = df[df["config"] == config]
        plt.plot(
            dados["n_documentos_reais"],
            dados[coluna],
            marker="o",
            linewidth=2,
            label=config.upper(),
        )

    plt.title(titulo)
    plt.xlabel("Número real de documentos")
    plt.ylabel(ylabel)
    if escala == "log":
        plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(caminho, dpi=300)
    plt.close()


def main():
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "results" / "raw" / "runs.csv"
    fig_dir = base_dir / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Lendo os resultados...")
    df = pd.read_csv(csv_path)
    mediana_df = (
        df.groupby(["config", "n_documentos_reais"])[["t_consulta_ms", "n_comparacoes"]]
        .median()
        .reset_index()
        .sort_values(["config", "n_documentos_reais"])
    )

    configs = ["c1", "c2", "c3", "c4"]
    configs_comparacoes = ["c1", "c2", "c3"]
    gerar_grafico(
        mediana_df,
        configs,
        "t_consulta_ms",
        "Desempenho: Tempo de Consulta vs Tamanho do Corpus",
        "Tempo de Consulta (ms) - Mediana",
        "linear",
        fig_dir / "tempo_consulta_vs_n_linear.png",
    )
    gerar_grafico(
        mediana_df,
        configs,
        "t_consulta_ms",
        "Desempenho: Tempo de Consulta vs Tamanho do Corpus",
        "Tempo de Consulta (ms) - Mediana",
        "log",
        fig_dir / "tempo_consulta_vs_n_log.png",
    )
    gerar_grafico(
        mediana_df,
        configs_comparacoes,
        "n_comparacoes",
        "Esforço Computacional: Comparações vs Tamanho do Corpus",
        "Total de Comparações (Busca + Ordenação)",
        "linear",
        fig_dir / "comparacoes_vs_n_linear.png",
    )
    gerar_grafico(
        mediana_df,
        configs_comparacoes,
        "n_comparacoes",
        "Esforço Computacional: Comparações vs Tamanho do Corpus",
        "Total de Comparações (Busca + Ordenação)",
        "log",
        fig_dir / "comparacoes_vs_n_log.png",
    )

    print(f"Sucesso! Gráficos salvos em: {fig_dir}")


if __name__ == "__main__":
    main()