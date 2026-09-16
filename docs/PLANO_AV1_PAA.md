# Plano de Execução — Atividade 1 (AV1) de PAA — versão enxuta

**Equipe:** 5 integrantes
**Corpus:** *News of the Site Folha de S.Paulo* (Kaggle, `marlesson/news-of-the-site-folhauol`, `articles.csv`)
**Função de relevância:** BM25 (Okapi)
**Unidade de recuperação:** o artigo completo (título + texto), sem chunking

> **Princípio deste plano:** fazer o mínimo que cumpre integralmente o enunciado, e fazer bem feito. Tudo que não é exigido foi cortado.

**Decisão que remove muito trabalho:** não usar as 167.053 notícias. Trabalhem com duas amostras aleatórias fixas (seed fixa) de **5.000 e 50.000** documentos. O enunciado exige "pelo menos dois tamanhos de corpus" — isso basta, elimina o risco de estouro de memória e torna as 12 execuções rápidas. Registrem a decisão como limitação no relatório.

---
## 1. Ficha do corpus

Em `docs/FICHA_CORPUS.md`:

| Campo            | Conteúdo                                                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Nome / fonte     | News of the Site Folha de S.Paulo — Kaggle, autor `marlesson`                                                                               |
| URL              | `https://www.kaggle.com/datasets/marlesson/news-of-the-site-folhauol`                                                                       |
| Licença          | **Copiar literalmente da página do Kaggle.** Anotar também que o conteúdo jornalístico permanece sob direitos da Folha                      |
| Data de acesso   | Data real do download                                                                                                                       |
| Nº de documentos | 167.053 no total (jan/2015 – set/2017); **amostras usadas: 5.000 e 50.000**                                                                 |
| Idioma           | Português brasileiro                                                                                                                        |
| Formato          | CSV, colunas `title`, `text`, `date`, `category`, `subcategory`, `link`                                                                     |
| Limpeza          | Remoção de linhas com `text` vazio; remoção de duplicatas exatas de título+texto                                                            |
| Chunking         | Não utilizado — cada linha já é uma unidade curta e autocontida                                                                             |
| Riscos de viés   | Fonte única, linha editorial própria, cobertura concentrada em Sudeste/Brasília, período encerrado em 2017                                  |
| Limitações       | Não generaliza para corpora multilíngues ou de documentos longos; BM25 é puramente lexical; amostra de 50k não representa o corpus completo |

**Repositório:** `data/raw/*.csv` no `.gitignore`. Versionar apenas `data/sample/sample_5k.csv` e o script `scripts/download_corpus.py`.

---

## 2. Parte A — Especificação e corretude

### 2.1 Definição formal

**Entrada:** corpus `C = {d₁, …, d_N}`; consulta `q` com `m` termos normalizados; inteiro `k ≥ 1`; parâmetros `k₁ = 1.2`, `b = 0.75`.

**Saída:** lista `R` com até `k` documentos de `C`.

**Função de relevância (BM25):**

```
score(q, d) = Σ_{t ∈ q}  IDF(t) · [ f(t,d) · (k₁ + 1) ]
                         ─────────────────────────────────────────
                         f(t,d) + k₁ · (1 − b + b · |d| / avgdl)

IDF(t) = ln( (N − n_t + 0.5) / (n_t + 0.5) + 1 )
```

`f(t,d)` = frequência do termo no documento; `|d|` = nº de tokens; `avgdl` = comprimento médio; `n_t` = nº de documentos que contêm `t`.

> Usem essa forma de IDF (a do Lucene, com o `+1` interno). A versão clássica fica **negativa** quando `n_t > N/2`, e num corpus de notícias em português palavras como "ano" e "brasil" caem nesse caso — o termo passaria a *reduzir* o score de quem o contém. Uma frase no relatório explicando essa escolha já rende ponto em corretude, e custa nada.

**Pré-condições:** `C ≠ ∅`; `avgdl > 0`; `k ≥ 1`; índice construído sobre o mesmo `C` da consulta.

**Pós-condições:**
- `|R| ≤ min(k, |{d ∈ C : score(q,d) > 0}|)`
- `R ⊆ C`, sem repetições
- `score(q, rᵢ) ≥ score(q, rᵢ₊₁)` para todo `i`
- Desempate determinístico: ordena de forma crescente por id quando há empate no score
- A saída é idêntica nas três configurações para a mesma `(q, k, C)`

**Casos de borda (viram os testes de `tests/`):**
Os casos abaixo estão organizados em Condição -> Efeito
1. Consulta (query) vazia após normalização -> `R = ⟨⟩`
2. Todos os termos fora do vocabulário -> `R = ⟨⟩`
3. `k > N` -> retorna todos os documentos com score positivo
4. Empate de scores -> verificar determinismo pelo `doc_id`
5. Documento com texto vazio -> sem divisão por zero
6. Termo repetido na consulta -> definir e documentar o comportamento

Seis testes são suficientes.

### 2.2 Normalização

Cinco passos, sem discussão longa:
1. Concatenar `title + " " + text`;
2. Lowercase;
3. Remover acentos (Unicode NFD);
4. Tokenizar com regex `\w+`;
5. Remover stopwords do português (NLTK).

> Implementado em `src/normalize.py`: as stopwords vêm de `nltk.corpus.stopwords.words("portuguese")` (~200 palavras), baixadas automaticamente na primeira execução (`nltk.download("stopwords")`, mesmo padrão do `kagglehub` para o corpus). As palavras são normalizadas (lowercase + sem acento) antes de entrar no filtro, para baterem com os tokens — que já passaram pelo mesmo `strip_accents()`. Se não houver rede, cai para uma lista manual pequena de fallback, só para não travar o pipeline; registrar isso como limitação se algum experimento rodar nesse modo.

### 2.3 Algoritmos — o conjunto mínimo

Nós vamos implementar:

| Exigência | Implementação da equipe |
|---|---|
| Busca linear | Varredura dos documentos calculando o score BM25 (`linear_search.py`, C1) |
| Estratégia de ordenação | Insertion Sort (`sorting.py`, C1 e C2) |
| Busca binária ou indexada | Índice invertido com hashmap: termo → posting list (`inverted_index.py`, C2 e C3) |
| Divisão e conquista | Merge Sort (`sorting.py`, C3) |
| Comparação com biblioteca | `rank_bm25` (BM25Okapi) e `sorted()` do Python, usados como baseline externo (C4) |

O índice invertido será construído uma vez por corpus apenas para C2 e C3. Cada posting list armazenará pares `(doc_id, frequência_do_termo)`. Durante a consulta, o hashmap será consultado diretamente para obter as posting lists dos termos da query; os documentos que não aparecem nessas listas não serão examinados.

Manteremos uma interface comum em `pipeline.py`, para que as etapas de experimentos, métricas e avaliação possam reutilizar os mesmos resultados sem depender dos detalhes internos de cada algoritmo.

### 2.4 Corretude

Deve-se provar a corretude de um algoritmo implementado, utilizando:

- **Invariante de laço**, ou
- **prova por indução**, ou
- **argumento formal de pré e pós condições**

A justificativa deve conter claramente:
- O que se deseja provar;
- Hipóteses;
- Inicialização, manutenção e término do invariante (se adotado a prova por invariante de laço);
- Limites do argumento;
- Casos em que a prova não se aplica

**Sugestão**: Provar **Merge Sort** utilizando **prova por indução**.

---

## 3. Parte B — Análise assintótica

Nós vamos analisar pelo menos dois dos algoritmos implementados. Para orientar essa parte, podemos usar os seguintes parâmetros:

- `N`: número de documentos;
- `L`: total de tokens do corpus;
- `m`: número de termos da consulta;
- `k`: número de resultados;
- `n_t`: tamanho da posting list de um termo.

Também precisamos registrar as operações elementares observáveis no código, como comparações, acessos a listas, consultas ao hashmap, atribuições e chamadas recursivas. A análise deve distinguir o custo da ingestão, da construção do índice, da consulta e da ordenação, além de comentar fatores que não aparecem no modelo RAM, como leitura do CSV, cache, Python e hashing.

Para a parte de recorrências, podemos analisar o Merge Sort e explicar a origem de cada termo diretamente nas chamadas recursivas e na etapa de intercalação. Como o índice escolhido usa hashmap, não teremos uma recorrência de busca binária para analisar. A análise detalhada, os casos melhor/pior/médio e as soluções das recorrências ficam para a equipe completar no relatório.

---

## 4. Parte C — Experimentos

### 4.1 Três configurações

| Configuração | O que muda | Composição |
|---|---|---|
| **C1 — baseline** | referência | Busca linear em todos os documentos + Insertion Sort |
| **C2 — indexada** | troca a busca | Índice invertido com hashmap + Insertion Sort apenas sobre os candidatos |
| **C3 — divisão e conquista** | troca a ordenação | Índice invertido com hashmap + Merge Sort |
| **C4 — baseline externo** | referência | `rank_bm25` + `sorted()` |

Com esse desenho, nós isolamos uma variável por vez: C1→C2 compara a busca linear com a busca indexada, enquanto C2→C3 compara as estratégias de ordenação sobre a mesma busca indexada.

**C4 não entra na matriz de execução cronometrada da seção 4.2** — ela usa uma biblioteca de terceiros (item 5 da seção 5.2 do enunciado) e serve só de baseline de qualidade para o Precision@5 (seção 4.4). Quem quiser cronometrá-la também pode, mas as 18 execuções obrigatórias são só C1+C2+C3.

#### Interface para os experimentos: `src/pipeline.py`

As quatro configurações já estão implementadas e prontas para uso em `src/pipeline.py`, com uma interface única para quem for medir tempo/memória (`experiments/run_experiments.py`) ou calcular Precision@5:

```python
from ingest import ensure_csv
from normalize import clean_text, tokenize
from pipeline import build_corpus, build_corpus_index, run

# uma vez, fora de qualquer bloco cronometrado:
csv_path = ensure_csv("articles.csv")

# uma vez por configuração e tamanho de N:
config = "c4"
corpus = build_corpus(
    csv_path, sample_size=5000, seed=42, config=config
)

# necessário apenas para C2 e C3:
if config in ("c2", "c3"):
    build_corpus_index(corpus)

# uma vez por (configuração, consulta, k):
query_tokens = tokenize(clean_text("impeachment de Dilma Rousseff"))
result = run(config, corpus, query_tokens, k=5)
```

Pontos importantes para quem for cronometrar (`experiments/run_experiments.py`):

- **`ensure_csv()` deve ser chamado antes da medição.** Se o CSV não estiver disponível localmente, essa função pode fazer download. O tempo de rede não deve entrar em `t_ingestao_ms`. Depois que o caminho for resolvido, a medição deve começar antes de `build_corpus(csv_path, ...)`.
- **Os imports devem ocorrer antes da medição.** O módulo `normalize.py` carrega as stopwords do NLTK durante o import e pode baixar o recurso na primeira execução. O script de experimentos deve importar os módulos e garantir os recursos antes de iniciar `time.perf_counter()`.
- `build_corpus_index(corpus)` só para C2/C3. `build_corpus()` não constrói o índice invertido — C1 e C4 nunca o usam, então não pagam esse custo. Para C4, passe `config="c4"` a `build_corpus()`; assim o `BM25Okapi` já estará pronto antes da primeira consulta. Chamar `run("c2"|"c3", ...)` sem antes construir o índice lança `ValueError` de propósito, para detectar o erro cedo.
- `corpus` pode ser reaproveitado entre as consultas e as repetições de um mesmo `(configuração, N)`. Só refaçam `build_corpus()` e, para C2/C3, `build_corpus_index()` quando `N` mudar.
- `BM25Okapi` é importado no início de `pipeline.py` e construído durante `build_corpus(..., config="c4")`. Assim, a primeira chamada a `run("c4", ...)` não tem custo de inicialização. O tempo de construção deve ser registrado na preparação de C4, separado do tempo das consultas.

`run(config, corpus, query_tokens, k)` aceita `config ∈ {"c1", "c2", "c3", "c4"}` e devolve um `ExecutionResult` com:

| Campo | Conteúdo |
|---|---|
| `results` | `list[(doc_id, score)]`, já ordenado e cortado em `k` |
| `search_comparisons` | operações contadas durante a busca |
| `sort_comparisons` | comparações feitas pelo Insertion Sort ou Merge Sort |
| `candidate_count` | número de documentos com score positivo antes do corte por `k` |

`pipeline.py` não mede nada sozinho, só executa. Também funciona como CLI (`python pipeline.py --config c2 --n 5000 --k 5 --query "..."`) para testes manuais rápidos — a CLI resolve o CSV antes de construir o corpus e constrói o índice quando necessário.

### 4.2 Matriz de execução

- **Tamanhos:** `N ∈ {5.000, 50.000}` — dois, o mínimo exigido
- **Consultas:** **5 fixas**, ex.: *"impeachment de Dilma Rousseff"*, *"operação Lava Jato delação"*, *"reforma da previdência"*, *"crise hídrica em São Paulo"*, *"eleição municipal 2016"*
- **`k` fixo em 5** — sem experimento de variação de `k`
- **Repetições:** 3 por cenário (1 aquecimento descartado)
- **Total: 3 configurações × 2 tamanhos × 3 repetições = 18 execuções**, acima do mínimo de 12

### 4.3 Métricas (uma linha por execução em `results/raw/runs.csv`)

Para cada execução, a seção 7.3 da atividade pede que calculemos: `t_ingestao_ms, t_indice_ou_ordenacao_ms, t_consulta_ms, t_total_ms, memoria_pico_mb, n_comparacoes, n_resultados, precision_at_5, resultado_vazio`

Seria interessante também relacionar cada uma dessas métricas com `config, N, k, query_id, repeticao`, de forma que fique evidente a qual run as métricas se referem.

Sugestões de como calcular as métricas:
- **Tempo:** `time.perf_counter()`, reportar a mediana das 3 repetições
- **Memória:** `tracemalloc.get_traced_memory()` — só isso, sem `psutil`
- **Comparações/operações:** `result.search_comparisons + result.sort_comparisons` (ambos já vêm contados por `src/pipeline.py`, sem precisar de contador manual). Para a busca indexada, a contagem inclui o acesso ao hashmap e as operações realizadas ao percorrer as posting lists.
- **Ambiente:** `pip freeze > results/requirements-lock.txt` e um `results/environment.txt` com SO, versão do Python, CPU e RAM

### 4.4 Precision@5 — versão simples

1. Para cada uma das 5 consultas, juntar o top-5 de C1, C2, C3 e do `rank_bm25` (≈8–15 documentos únicos após deduplicação).
2. Cada documento do pool é julgado relevante/não relevante por **dois integrantes**; divergência resolvida por conversa entre os dois.
3. Salvar em `data/qrels.csv` (`query_id, doc_id, relevante`).

São cerca de 50 julgamentos no total, uma hora de trabalho da equipe. Sem cálculo de concordância entre anotadores — não é exigido.

> Nota metodológica de uma linha no relatório: C1, C2 e C3 devem produzir **exatamente o mesmo ranking**, pois usam a mesma função de relevância. O `Precision@5` mede a qualidade do BM25, não a diferença entre as configurações — o que as diferencia é tempo e memória. Deixar isso explícito evita a interpretação errada dos resultados na banca.

### 4.5 Gráficos

Apesar do mínimo exigido ser um, vamos fazer dois gráficos:

1. **Tempo de consulta × N** — uma curva por configuração
2. **Nº de comparações × N** — mesma estrutura

 Esse par de gráficos será o argumento central do trabalho: se as duas curvas tiverem o mesmo formato, a análise do modelo RAM prevê o comportamento real; onde divergirem, teremos o material da discussão sobre fatores fora do modelo RAM.

---

## 5. Parte D — Relação com IA generativa

Seção de texto, sem código adicional. Sugestão para os sete pontos pedidos pela seção 8 da atividade:

1. **Incorporação ao prompt:** template `[n] título — data — categoria — texto — link`, com instrução ao modelo para citar os índices usados.
2. **Impacto de `k` e do tamanho do documento:** calcular a média e o máximo de tokens por artigo nas amostras (duas linhas de código). Sem chunking, a unidade é a notícia inteira, então o custo de contexto com `k = 5` varia bastante conforme quais artigos são recuperados. É a justificativa honesta da decisão de não usar chunking, e também a explicação de por que chunking existe.
3. **Léxico vs. semântico:** BM25 falha em *vocabulary mismatch*. Exemplo concreto do corpus: a consulta "alta do dólar" não recupera artigos que dizem apenas "câmbio" ou "moeda americana". Basta mostrar um caso real encontrado nos experimentos.
4. **Riscos de recuperação:** fonte única (só Folha), viés editorial e — o mais forte aqui — **desatualização**: o acervo termina em set/2017, então uma pergunta sobre o presente recuperaria documentos coerentes e obsoletos.
5. **Afirmações não sustentadas:** com documentos só parcialmente relevantes, o LLM tende a preencher lacunas. Mitigação: instrução de abstenção e limiar mínimo de score BM25 — vocês já têm esse limiar de graça, pois score 0 é bem definido.
6. **Rastreabilidade:** o campo `link` do dataset dá URL verificável por artigo; incluir score, data e link junto de cada documento no prompt.
7. **Trade-offs:** parágrafo relacionando precisão, latência, memória e custo de geração, usando os números reais dos experimentos.

---
## 6. Estrutura do repositório

```
PAA_UFS_2026_2_.../
├─ README.md              # deps, ambiente, comandos, reprodução, "Vídeo da atividade"
├─ VIDEO.md               # URL, data de gravação, participantes
├─ LICENSE
├─ .gitignore             # data/raw/*.csv
├─ requirements.txt
├─ docs/
│  ├─ FICHA_CORPUS.md
│  ├─ DECLARACAO_IA.md
│  ├─ CONTRIBUICOES.md
│  └─ relatorio.pdf
├─ src/
│  ├─ ingest.py           # CSV, limpeza, amostragem
│  ├─ normalize.py        # lowercase, acentos, tokenização, stopwords
│  ├─ bm25.py             # scoring
│  ├─ linear_search.py    # C1
│  ├─ inverted_index.py   # hashmap termo → posting list (C2, C3)
│  ├─ sorting.py          # insertion_sort, merge_sort + contador
│  └─ pipeline.py         # CLI: --config --N --k --query
├─ experiments/
│  ├─ run_experiments.py
│  └─ make_plots.py
├─ tests/test_edge_cases.py
├─ data/
│  ├─ sample/sample_5k.csv
│  ├─ queries.csv
│  └─ qrels.csv
├─ results/
│  ├─ raw/runs.csv
│  ├─ environment.txt
│  └─ figures/
└─ scripts/download_corpus.py
```

Sete arquivos em `src/`, nenhum deles longo. É o suficiente.
