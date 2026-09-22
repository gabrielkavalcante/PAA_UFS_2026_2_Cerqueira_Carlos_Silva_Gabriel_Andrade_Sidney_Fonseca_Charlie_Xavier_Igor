# Protótipo de Recuperação de Contexto para IA Generativa (BM25)

> **Projeto e Análise de Algoritmos (PAA) — 2026.2**  
> **Universidade Federal de Sergipe (UFS)** — Departamento de Computação  
> **Atividade 1 (AV1):** Corretude, eficiência e recuperação de contexto para IA generativa  
> **Tema 6:** Notícias de Fontes Públicas (Folha de S.Paulo / UOL)

---

## Equipe e Atribuições Individuais

* **Gabriel Cavalcante Silva** *(Engenharia de Dados e Infraestrutura)*: Configuração do repositório, scripts de ingestão e normalização do corpus, e estruturação inicial do relatório técnico em LaTeX.
* **Carlos Adriano Gama Cerqueira** *(Engenharia de Software: Algoritmos)*: Implementação da busca linear, dos algoritmos de ordenação (*Insertion Sort* e *Merge Sort*) e da busca indexada por tabela hash.
* **Sidney Matheus Santos de Andrade** *(Ciência da Computação: Teoria)*: Justificativa formal de corretude (invariante de laço e indução), análise no modelo RAM (melhor, pior e caso médio) e recorrências matemáticas.
* **Igor Rafael Silva Xavier** *(Análise de Desempenho: Testes)*: Condução dos experimentos de escalabilidade temporal e espacial, coleta de métricas via `tracemalloc` e geração dos gráficos comparativos.
* **Charlie Rodrigues Fonseca** *(Análise Qualitativa e Documentação)*: Implementação dos scripts de avaliação de tokens e Precision@5, detecção de *vocabulary mismatch*, condução da anotação humana de relevância, elaboração do `README.md`, `VIDEO.md` e redação das seções correspondentes no relatório em LaTeX.

---

## Vídeo da atividade

Conforme exigido na **Seção 11 do edital**, o vídeo de apresentação da equipe com duração máxima de 10 minutos está disponível no link abaixo:

* **URL do Vídeo:** `[URL PENDENTE DE GRAVAÇÃO - INSERIR LINK DO YOUTUBE/DRIVE PÚBLICO]`
* **Arquivo complementar de metadados:** Consulte [`VIDEO.md`](VIDEO.md) para a relação detalhada dos tópicos apresentados e divisão das falas entre os integrantes.

---

## Visão Geral do Protótipo

O projeto implementa e avalia comparativamente quatro configurações de recuperação de documentos relevantes baseadas no modelo probabilístico **BM25**, investigando os trade-offs entre custo computacional (tempo e memória) e qualidade semântica da recuperação para sistemas de Geração Aumentada por Recuperação (*Retrieval-Augmented Generation* — RAG):

| Configuração | Estratégia de Busca | Algoritmo de Ordenação | Complexidade Teórica (Consulta) |
| :--- | :--- | :--- | :---: |
| **C1** | Busca Linear (percorre todo o corpus) | *Insertion Sort* | $\Theta(N \cdot m + N^2)$ |
| **C2** | Índice Invertido (tabela hash `dict`) | *Insertion Sort* (apenas candidatos $C$) | $\Theta(m + P + C^2)$ |
| **C3** | Índice Invertido (tabela hash `dict`) | *Merge Sort* (apenas candidatos $C$) | $\Theta(m + P + C \log C)$ |
| **C4** | Baseline externo (`rank_bm25`) | `sorted()` nativo do Python (Timsort) | Baseline de biblioteca |

*Onde $N$ é o tamanho do corpus, $m$ é o número de tokens da consulta, $P$ é a soma das posting lists acessadas e $C$ é o total de candidatos distintos ($C \ll N$).*

---

## Obtenção e Acesso ao Corpus

O acervo utilizado reúne notícias públicas em português do portal Folha de S.Paulo / UOL (período de 2015 a 2017). Para cumprir as diretrizes do edital quanto ao versionamento de dados volumosos no Git, o arquivo bruto `articles.csv` não é versionado diretamente e pode ser obtido de duas formas:

1. **Download Automatizado (Recomendado):** O script `src/ingest.py` baixa o dataset automaticamente da plataforma Kaggle na primeira execução utilizando a biblioteca `kagglehub`.
2. **Download Direto (Google Drive da Equipe):** [Acessar Corpus articles.csv no Google Drive](https://drive.google.com/file/d/1KfPinMM6bRGGy5FNvC_DWqFoAJJpXYBu/view?usp=sharing). Após o download, coloque o arquivo `articles.csv` na raiz do projeto ou dentro de `data/`.

---

## Estrutura do Repositório

```text
├── data/
│   ├── queries.csv              # As 5 consultas padronizadas de teste (q1 a q5)
│   └── pool_judgments.csv       # Pool deduplicado de 132 artigos para anotação de relevância
├── docs/                        # Documentação auxiliar e especificações
├── especificacoes.md            # Edital oficial completo da Atividade 1 (AV1)
├── experiments/
│   ├── run_experiments.py       # Medição de tempo, memória e contagem de comparações (300 runs)
│   ├── make_plots.py            # Geração dos gráficos comparativos (linear e logarítmico)
│   ├── quality_analysis.py      # Extração estatística de tokens (médias, quartis, P95) e mismatch
│   └── compute_precision.py     # Cálculo automatizado de Precision@5 a partir do pool
├── pyproject.toml               # Metadados do projeto e dependências oficiais
├── README.md                    # Este arquivo com instruções completas de reprodução
├── results/
│   └── raw/                     # Dados brutos em CSV gerados pelos experimentos
│       ├── runs.csv             # 300 execuções cronometradas com métricas de tempo e memória
│       ├── token_stats.csv      # Estatísticas de tokens por tamanho de corpus (médias e percentis)
│       ├── vocab_mismatch.csv   # Análise sistemática de vocabulary mismatch por consulta
│       ├── precision_at_5.csv   # Precision@5 detalhada por consulta e configuração
│       └── precision_summary.csv# Resumo macro de Precision@5 por configuração e tamanho N
├── src/                         # Implementação dos algoritmos clássicos e pipeline
│   ├── bm25.py                  # Fórmula do BM25 (TF, IDF e parâmetros k1=1.5, b=0.75)
│   ├── ingest.py                # Download, amostragem reprodutível (seed 42) e preparação
│   ├── inverted_index.py        # Construção e consulta do índice invertido (tabela hash)
│   ├── linear_search.py         # Busca linear sobre todos os documentos (Configuração C1)
│   ├── normalize.py             # Normalização (lowercase, remoção de acentos e stopwords NLTK)
│   ├── pipeline.py              # Orquestrador do pipeline de busca e ranking (C1, C2, C3, C4)
│   └── sorting.py               # Insertion Sort e Merge Sort instrumentados com comparador estável
├── VIDEO.md                     # Metadados obrigatórios, checklist e roteiro do vídeo
└── .gitignore                   # Regras de exclusão para CSVs pesados e artefatos de build
```

---

## Ambiente e Dependências

O projeto foi desenvolvido e testado em ambiente Linux (Python 3.13+ / 3.14). 

### Pré-requisitos
* Python 3.13 ou superior (compatível com 3.12)
* Gerenciador de pacotes [`uv`](https://github.com/astral-sh/uv) (recomendado) ou `pip` padrão.

### Instalação

#### Opção A: Utilizando `uv` (Recomendado — Rápido e Determinístico)
```bash
# Sincroniza e instala as dependências a partir do pyproject.toml / uv.lock
uv sync
```

#### Opção B: Utilizando `venv` e `pip` tradicional
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Principais bibliotecas instaladas:
* `pandas>=3.0.5`: Manipulação tabular de dados
* `numpy>=2.0.0`: Cálculos numéricos e quantis estatísticos
* `nltk>=3.9.1`: Processamento de texto e stopwords em português
* `matplotlib>=3.11.2`: Geração de gráficos comparativos
* `rank-bm25>=0.2.2`: Baseline externo para a configuração C4
* `kagglehub>=1.0.2`: Ingestão automática do dataset

---

## Instruções de Reprodução dos Experimentos

Execute os comandos a partir da raiz do repositório:

### 1. Experimentos de Desempenho e Escalabilidade
Executa a grade completa de 300 corridas cronometradas ($N \in \{1.000, 5.000, 10.000, 25.000, 50.000\}$, 5 consultas, 3 repetições por configuração com aquecimento descartado):
```bash
python experiments/run_experiments.py
# Saída gerada: results/raw/runs.csv
```

### 2. Geração dos Gráficos Comparativos
Gera as figuras com os tempos de consulta e número de comparações em escalas linear e logarítmica:
```bash
python experiments/make_plots.py
# Gráficos salvos em: results/plots/ e relatorio_AV1_PAA__4_/
```

### 3. Análise de Tokens, Vocabulary Mismatch e Pool de Relevância
Calcula os comprimentos de texto (média, desvio padrão, quartis Q1/Q3, percentil 95% e máximo), identifica termos não correspondidos e consolida o pool de 132 documentos únicos para julgamento humano:
```bash
python experiments/quality_analysis.py
# Saídas geradas:
# - results/raw/token_stats.csv
# - results/raw/vocab_mismatch.csv
# - data/pool_judgments.csv
```

### 4. Cálculo de Precision@5
A partir das anotações humanas contidas na coluna `relevant` de `data/pool_judgments.csv`, computa a Precision@5 por consulta e médias macro:
```bash
python experiments/compute_precision.py
# Saídas geradas:
# - results/raw/precision_at_5.csv
# - results/raw/precision_summary.csv
```

---

## Resumo dos Resultados Experimentais

### 1. Desempenho e Comparações ($N = 49.741$ documentos reais)
* **C1 (Linear + Insertion):** $25.147{,}78\text{ ms}$ e $164.720.059$ comparações — confirma o gargalo quadrático $\Theta(N^2)$.
* **C2 (Índice Hash + Insertion):** $725{,}18\text{ ms}$ e $12.979.567$ comparações — aceleração expressiva ao ordenar apenas candidatos ($C \ll N$).
* **C3 (Índice Hash + Merge Sort):** $12{,}23\text{ ms}$ e $98.641$ comparações — velocidade **2.056 vezes superior à C1**, comprovando a eficiência assintótica de $\Theta(C \log C)$.
* **C4 (Baseline externo):** $40{,}99\text{ ms}$ com maior consumo de memória ($11{,}35\text{ MB}$ contra $0{,}82\text{ MB}$ de C3).

### 2. Qualidade da Recuperação e Ausência de Chunking
* **Precision@5:** Cresce de forma consistente com a escala do corpus ($0{,}76$ em $N=1.000 \to 0{,}96$ em $N=5.000 \to 1{,}00$ em $N \ge 10.000$), com empate absoluto entre C1, C2 e C3 conforme previsto teoricamente.
* **Estatísticas de Tokens:** A mediana mantém-se entre $238$ e $244$ tokens, com 75% dos artigos contendo até $341$ tokens (Q3) e 95% contendo até $566$ tokens (P95). Isso valida a decisão de **não fragmentar o corpus em chunks**, mantendo as notícias íntegras sem risco de estouro de janela em LLMs ($\approx 1.200\text{--}1.340$ tokens para $k=5$).

---

## Declaração de Uso de IA Generativa

O projeto utilizou apoio das ferramentas **Claude (Anthropic)** e **Google Antigravity (Claude Sonnet 4.6 Thinking e Gemini 3.8 Flash)** como assistentes técnicos para compreensão de requisitos, estruturação de scripts analíticos e formatação de texto. Todo o desenvolvimento foi realizado sob **supervisão humana estrita**, com revisão integral de código, provas matemáticas refeitas pela equipe e anotação 100% manual dos 132 documentos do pool de avaliação. Maiores detalhes constam na Seção 16 do Relatório Técnico.

---

## Licença

Este projeto é desenvolvido para fins exclusivamente acadêmicos no âmbito da Universidade Federal de Sergipe (UFS).
