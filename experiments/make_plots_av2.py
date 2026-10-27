"""Pacote 4 — Geração de results/raw/av2_resumo.csv e das 5 figuras da AV2."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent
RAW_DIR = RAIZ / "results" / "raw"
FIG_DIR = RAIZ / "results" / "figures"

METHODS = [
    "topk_av1",
    "greedy_score",
    "greedy_ratio",
    "greedy_ratio_plus",
    "dp",
]

METHOD_LABELS = {
    "topk_av1": "topk_av1 (k=5)",
    "greedy_score": "greedy_score",
    "greedy_ratio": "greedy_ratio",
    "greedy_ratio_plus": "greedy_ratio_plus",
    "dp": "dp (Mochila 0/1)",
}

METHOD_COLORS = {
    "topk_av1": "#7f7f7f",
    "greedy_score": "#ff7f0e",
    "greedy_ratio": "#2ca02c",
    "greedy_ratio_plus": "#9467bd",
    "dp": "#1f77b4",
}

METRICS_SPEC = [
    ("t_selecao_ms", "Tempo de seleção (ms)", "Tempo de Seleção por Orçamento W", "av2_tempo_selecao.png"),
    ("total_score", "Score BM25 total", "Valor Total (Score BM25) por Orçamento W", "av2_valor_total.png"),
    ("tokens_used", "Tokens usados", "Tokens Usados por Orçamento W", "av2_tokens_usados.png"),
    ("coverage", "Cobertura da consulta (0 a 1)", "Cobertura dos Termos da Consulta por Orçamento W", "av2_cobertura.png"),
    ("redundancy", "Redundância Jaccard (0 a 1)", "Redundância Média (Jaccard) por Orçamento W", "av2_redundancia.png"),
]


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa por method, N e W calculando média e desvio padrão."""
    metric_cols = [
        "t_selecao_ms",
        "memoria_pico_mb",
        "ops",
        "n_docs_sel",
        "tokens_used",
        "budget_ok",
        "total_score",
        "coverage",
        "redundancy",
    ]
    agg_dict: dict[str, tuple[str, str]] = {"n_runs": ("rep", "count")}
    for col in metric_cols:
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_std"] = (col, lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0)

    resumo = df.groupby(["N", "W", "method"], as_index=False).agg(**agg_dict)

    order_map = {m: i for i, m in enumerate(METHODS)}
    resumo["_order"] = resumo["method"].map(lambda m: order_map.get(m, 99))
    resumo = resumo.sort_values(["N", "W", "_order"]).drop(columns=["_order"])
    return resumo.round(6)


def plot_grouped_bars(resumo: pd.DataFrame, metric: str, ylabel: str, title: str, out_path: Path) -> None:
    """Barras agrupadas por W, uma cor por método: N=50.000 (esq.) e N=10.000 (dir.)."""
    available_n = sorted(resumo["N"].unique(), reverse=True)
    # Garante N=50.000 primeiro (esquerda) e N=10.000 ao lado (direita), se presentes
    if 50_000 in available_n and 10_000 in available_n:
        n_panels = [50_000, 10_000]
    else:
        n_panels = available_n

    fig, axes = plt.subplots(
        1, len(n_panels), figsize=(6.5 * len(n_panels), 5.0), sharey=True, squeeze=False
    )

    methods_present = [m for m in METHODS if m in resumo["method"].unique()]
    n_methods = max(len(methods_present), 1)
    bar_width = 0.8 / n_methods

    for idx_panel, n_val in enumerate(n_panels):
        ax = axes[0, idx_panel]
        sub_n = resumo[resumo["N"] == n_val]
        w_values = sorted(sub_n["W"].unique())
        x = np.arange(len(w_values))

        for idx_m, method in enumerate(methods_present):
            sub_m = sub_n[sub_n["method"] == method].set_index("W")
            means = [float(sub_m.loc[w, f"{metric}_mean"]) if w in sub_m.index else 0.0 for w in w_values]
            stds = [float(sub_m.loc[w, f"{metric}_std"]) if w in sub_m.index else 0.0 for w in w_values]

            offset = (idx_m - (n_methods - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                means,
                width=bar_width,
                yerr=stds,
                capsize=3,
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method, None),
                edgecolor="black",
                linewidth=0.5,
            )

        if metric == "tokens_used":
            for idx_w, w_val in enumerate(w_values):
                ax.hlines(
                    w_val,
                    idx_w - 0.42,
                    idx_w + 0.42,
                    colors="red",
                    linestyles="--",
                    linewidth=1.2,
                    label="Orçamento W" if idx_w == 0 else None,
                )

        panel_label = "Principal" if n_val == 50_000 else "Secundário"
        ax.set_title(f"N = {n_val:,}".replace(",", ".") + f" ({panel_label})")
        ax.set_xlabel("Orçamento de Tokens (W)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(w) for w in w_values])
        ax.grid(axis="y", linestyle=":", alpha=0.7)
        if idx_panel == 0:
            ax.set_ylabel(ylabel)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=min(len(labels), 6), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    runs_csv = RAW_DIR / "av2_runs.csv"
    if not runs_csv.exists():
        print(f"Erro: arquivo {runs_csv} não encontrado. Execute experiments/run_av2.py primeiro.")
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(runs_csv)
    resumo = build_summary(df)
    resumo_path = RAW_DIR / "av2_resumo.csv"
    resumo.to_csv(resumo_path, index=False)
    print(f"Tabela resumo salva em: {resumo_path}")

    for metric, ylabel, title, filename in METRICS_SPEC:
        plot_grouped_bars(resumo, metric, ylabel, title, FIG_DIR / filename)
        print(f"Figura gerada: {FIG_DIR / filename}")

    return 0


if __name__ == "__main__":
    sys.exit(main())