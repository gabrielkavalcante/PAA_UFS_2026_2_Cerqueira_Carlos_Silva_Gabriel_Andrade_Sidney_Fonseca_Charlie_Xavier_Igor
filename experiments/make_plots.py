import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Define os caminhos dos arquivos
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "results" / "raw" / "runs.csv"
    fig_dir = base_dir / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Lendo os resultados...")
    df = pd.read_csv(csv_path)

    mediana_df = df.groupby(['config', 'N'])[['t_consulta_ms', 'n_comparacoes']].median().reset_index()


    plt.figure(figsize=(10, 6))
    for config in ['c1', 'c2', 'c3', 'c4']:
        dados = mediana_df[mediana_df['config'] == config]
        plt.plot(dados['N'], dados['t_consulta_ms'], marker='o', linewidth=2, label=config.upper())

    plt.title('Desempenho: Tempo de Consulta vs Tamanho do Corpus')
    plt.xlabel('Número de Documentos (N)')
    plt.ylabel('Tempo de Consulta (ms) - Mediana')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_dir / 'tempo_consulta_vs_n.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))

    for config in ['c1', 'c2', 'c3']:
        dados = mediana_df[mediana_df['config'] == config]
        plt.plot(dados['N'], dados['n_comparacoes'], marker='o', linewidth=2, label=config.upper())

    plt.title('Esforço Computacional: Comparações vs Tamanho do Corpus')
    plt.xlabel('Número de Documentos (N)')
    plt.ylabel('Total de Comparações (Busca + Ordenação)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_dir / 'comparacoes_vs_n.png', dpi=300)
    plt.close()

    print(f"Sucesso! Gráficos salvos em: {fig_dir}")

if __name__ == "__main__":
    main()