import glob
import os
from typing import cast

import pandas as pd

from normalize import clean_text, tokenize


def ensure_csv(csv_path: str = "articles.csv") -> str:
    """Se o CSV já existir localmente, apenas devolve o caminho. Caso
    contrário, baixa o dataset do Kaggle via kagglehub e devolve o
    caminho do articles.csv dentro da pasta baixada."""
    if os.path.exists(csv_path):
        return csv_path

    import kagglehub

    print(f"'{csv_path}' nao encontrado. Baixando do Kaggle...")
    path = kagglehub.dataset_download("marlesson/news-of-the-site-folhauol")
    print("Path to dataset files:", path)

    candidate_paths = glob.glob(os.path.join(path, "*.csv"))
    if not candidate_paths:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {path}")

    # prefere um arquivo chamado exatamente articles.csv, se existir
    for candidate in candidate_paths:
        if os.path.basename(candidate) == "articles.csv":
            return candidate
    return candidate_paths[0]


def load_and_process_corpus(
    csv_path: str = "articles.csv", sample_size: int | None = 5000, seed: int = 42
) -> pd.DataFrame:
    """Lê o CSV, retira uma amostra fixa e reprodutível, limpa os dados
    e tokeniza os textos. Não há divisão em trechos: cada linha do CSV
    já é uma notícia curta e autocontida, então o documento inteiro é
    a unidade de recuperação."""

    print("Iniciando ingestao do CSV...")
    df = pd.read_csv(csv_path)

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    print("Limpando dados (linhas sem texto, duplicatas)...")
    df = df.dropna(subset=["text"])
    df = df.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)

    print("Normalizando textos (lowercase, sem acento, sem pontuacao)...")
    df["clean_title"] = df["title"].apply(clean_text)
    df["clean_text"] = df["text"].apply(clean_text)

    print("Tokenizando e removendo stopwords...")
    df["tokens"] = (df["clean_title"] + " " + df["clean_text"]).apply(tokenize)
    df["doc_id"] = df.index

    print(f"Processamento concluido. {len(df)} documentos (sem chunking).")
    return cast(pd.DataFrame, df[["doc_id", "title", "date", "link", "tokens"]])


if __name__ == "__main__":
    df = load_and_process_corpus()
    tokens_column = cast(pd.Series, df["tokens"])
    df.assign(tokens=tokens_column.apply(lambda t: " ".join(t))).to_csv(
        "corpus_normalizado.csv", index=False
    )
