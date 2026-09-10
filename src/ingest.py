import glob
import os
import re
import unicodedata
from typing import cast

import pandas as pd

STOPWORDS_PT: set[str] = {
    "a","o","as","os","de","da","do","das","dos","em","um","uma","uns","umas",
    "para","com","por","que","se","na","no","nas","nos","e","é","ao","aos",
    "à","às","como","mais","mas","ou","foi","ser","tem","têm","era","são",
    "sua","seu","suas","seus","este","esta","isso","isto","também","já",
    "entre","sobre","até","depois","antes","quando","onde","porque","não",
}  # fmt: skip


def garantir_csv(caminho_csv: str = "articles.csv") -> str:
    """Se o CSV ja existir localmente, so devolve o caminho. Caso
    contrario, baixa o dataset do Kaggle via kagglehub e devolve o
    caminho do articles.csv dentro da pasta baixada."""
    if os.path.exists(caminho_csv):
        return caminho_csv

    import kagglehub

    print(f"'{caminho_csv}' nao encontrado. Baixando do Kaggle...")
    path = kagglehub.dataset_download("marlesson/news-of-the-site-folhauol")
    print("Path to dataset files:", path)

    candidatos = glob.glob(os.path.join(path, "*.csv"))
    if not candidatos:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {path}")

    # prefere um arquivo chamado exatamente articles.csv, se existir
    for c in candidatos:
        if os.path.basename(c) == "articles.csv":
            return c
    return candidatos[0]


def remover_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def limpar_texto(texto: object) -> str:
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = remover_acentos(texto)
    texto = re.sub(r"[^\w\s]", "", texto)
    return texto


def tokenizar(texto_limpo: str) -> list[str]:
    tokens = texto_limpo.split()
    return [t for t in tokens if t not in STOPWORDS_PT]


def carregar_e_processar_corpus(
    caminho_csv: str = "articles.csv", n_amostra: int | None = 5000, seed: int = 42
) -> pd.DataFrame:
    """Le o CSV (baixando automaticamente do Kaggle se necessario),
    tira uma amostra fixa e reprodutivel, limpa e tokeniza.
    Sem chunking: cada linha do CSV ja e uma noticia curta e autocontida,
    entao o documento inteiro e a unidade de recuperacao."""
    caminho_csv = garantir_csv(caminho_csv)

    print("Iniciando ingestao do CSV...")
    df = pd.read_csv(caminho_csv)

    if n_amostra is not None and len(df) > n_amostra:
        df = df.sample(n=n_amostra, random_state=seed).reset_index(drop=True)

    print("Limpando dados (linhas sem texto, duplicatas)...")
    df = df.dropna(subset=["text"])
    df = df.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)

    print("Normalizando textos (lowercase, sem acento, sem pontuacao)...")
    df["titulo_limpo"] = df["title"].apply(limpar_texto)
    df["texto_limpo"] = df["text"].apply(limpar_texto)

    print("Tokenizando e removendo stopwords...")
    df["tokens"] = (df["titulo_limpo"] + " " + df["texto_limpo"]).apply(tokenizar)
    df["doc_id"] = df.index

    print(f"Processamento concluido. {len(df)} documentos (sem chunking).")
    return cast(pd.DataFrame, df[["doc_id", "title", "date", "link", "tokens"]])


if __name__ == "__main__":
    df = carregar_e_processar_corpus()
    tokens_col = cast(pd.Series, df["tokens"])
    df.assign(tokens=tokens_col.apply(lambda t: " ".join(t))).to_csv(
        "corpus_normalizado.csv", index=False
    )
