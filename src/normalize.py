"""Normalizacao de texto: lowercase, remocao de acentos, tokenizacao e
remocao de stopwords. Usado tanto na ingestao do corpus (ingest.py)
quanto na normalizacao da consulta (pipeline.py)."""

import re
import unicodedata

import nltk

# Lista manual de fallback, usada apenas se o download do corpus do NLTK falhar
_FALLBACK_STOPWORDS: set[str] = {
    "a","o","as","os","de","da","do","das","dos","em","um","uma","uns","umas",
    "para","com","por","que","se","na","no","nas","nos","e","ao","aos",
    "como","mais","mas","ou","foi","ser","tem","era","sao",
    "sua","seu","suas","seus","este","esta","isso","isto","tambem","ja",
    "entre","sobre","ate","depois","antes","quando","onde","porque","nao",
}  # fmt: skip


def strip_accents(text: str) -> str:
    """Remove os acentos (diacríticos) de um texto."""
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _load_stopwords() -> set[str]:
    """Carrega as stopwords do português a partir do NLTK, baixando o
    corpus automaticamente na primeira execução. As palavras são
    normalizadas (minúsculas e sem acento) para corresponderem aos
    tokens do corpus, que já passaram por strip_accents(). Se o
    download falhar (ambiente sem rede), usa a lista manual de
    fallback."""
    try:
        from nltk.corpus import stopwords as nltk_stopwords

        try:
            words = nltk_stopwords.words("portuguese")
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = nltk_stopwords.words("portuguese")
        return {strip_accents(word.lower()) for word in words}
    except (LookupError, OSError):
        print(
            "Aviso: nao foi possivel carregar as stopwords do NLTK "
            "(sem rede?). Usando lista manual de fallback."
        )
        return set(_FALLBACK_STOPWORDS)


PORTUGUESE_STOPWORDS: set[str] = _load_stopwords()


def clean_text(text: object) -> str:
    """Normaliza um texto: converte para minúsculas, remove acentos e
    remove a pontuação."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = strip_accents(text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def tokenize(cleaned_text: str) -> list[str]:
    """Divide um texto já normalizado em tokens e remove as
    stopwords."""
    tokens = cleaned_text.split()
    return [token for token in tokens if token not in PORTUGUESE_STOPWORDS]
