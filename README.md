\section{Algoritmos e Pseudocódigo}

\subsection{Busca Linear (Configuração C1)}
\begin{verbatim}
função busca_linear(consulta_tokens, corpus):
    resultados ← lista vazia
    para cada documento d em corpus:
        score ← score_bm25(consulta_tokens, d)
        resultados.adicionar((d.id, score))
    retornar resultados
\end{verbatim}
Complexidade: percorre os $N$ documentos do corpus, calculando o score BM25 de cada um contra os $m$ termos da consulta.

\subsection{Insertion Sort}
\begin{verbatim}
função insertion_sort(pares):
    para j de 1 até n-1:
        chave ← pares[j]
        i ← j - 1
        enquanto i ≥ 0 e vem_depois(pares[i], chave):
            pares[i+1] ← pares[i]
            i ← i - 1
        pares[i+1] ← chave
    retornar pares
\end{verbatim}
O critério \texttt{vem\_depois} implementa a ordem total definida na Seção~6: score decrescente, com desempate por \texttt{doc\_id} crescente. O algoritmo mantém um contador de comparações, utilizado na Seção~14 (Resultados) para amarrar a análise assintótica à medição experimental.

\subsection{Índice Invertido e Busca Binária (Configurações C2 e C3)}
\begin{verbatim}
função construir_indice_invertido(corpus):
    dicionário ← tabela vazia
    para cada documento d em corpus:
        para cada termo t em d:
            dicionário[t].adicionar(d.id)
    ordenar as chaves de dicionário
    retornar dicionário

função buscar_termo(dicionário, termo):
    retornar busca_binária(chaves_ordenadas(dicionário), termo)
\end{verbatim}
A busca binária localiza a posting list do termo em $\Theta(\log V)$, em vez de percorrer todo o corpus.

\subsection{Merge Sort (Configuração C3)}
\begin{verbatim}
função merge_sort(pares):
    se |pares| ≤ 1: retornar pares
    meio ← |pares| / 2
    esquerda ← merge_sort(pares[0:meio])
    direita ← merge_sort(pares[meio:])
    retornar mesclar(esquerda, direita)
\end{verbatim}

\section{Justificativa de Corretude}

Apresenta-se a prova de corretude do Insertion Sort, por invariante de laço.

\textbf{O que se quer provar:} ao final da execução, a lista de pares (\texttt{doc\_id}, \texttt{score}) está ordenada de forma não crescente por score e é uma permutação da entrada.

\textbf{Hipóteses:} o comparador implementa uma ordem total (score decrescente, desempate por \texttt{doc\_id} crescente); os scores são valores finitos.

\textbf{Invariante de laço:} no início de cada iteração do laço externo com índice $j$, o subvetor $A[1..j-1]$ contém os mesmos elementos que continha originalmente, agora em ordem não crescente de score.

\begin{itemize}
    \item \textbf{Inicialização:} $j = 2$, logo $A[1..1]$ tem um único elemento e está trivialmente ordenado.
    \item \textbf{Manutenção:} o laço interno desloca à direita todos os elementos de $A[1..j-1]$ que devem vir depois de $A[j]$ segundo o comparador, inserindo $A[j]$ na posição correta. Nenhum elemento é criado ou perdido, e o subvetor $A[1..j]$ permanece ordenado.
    \item \textbf{Terminação:} o laço termina com $j = n+1$; pelo invariante, $A[1..n]$ está ordenado e é permutação da entrada. $\blacksquare$
\end{itemize}

\textbf{Limites do argumento:} a prova assume aritmética exata. Em ponto flutuante IEEE-754, dois documentos podem receber scores que diferem no último bit conforme a ordem de acumulação dos termos da consulta; por isso, a ordem de iteração sobre os termos é fixada na implementação. A prova cobre apenas a corretude da ordenação, não a corretude do cálculo do score BM25 em si.

\section{Análise no Modelo RAM}

\textbf{Parâmetros:} $N$ (documentos do corpus), $L$ (total de tokens do corpus), $m$ (termos da consulta), $k$ (resultados retornados), $V$ (tamanho do vocabulário), $n_t$ (tamanho da posting list do termo $t$).

\textbf{Operações elementares consideradas:} comparação de scores, acesso indexado a vetor, lookup em tabela hash, atribuição, chamada recursiva.

\textbf{Fatores fora do modelo RAM:} leitura do CSV a partir do disco, localidade de cache, overhead do interpretador Python e do coletor de lixo, custo de hashing de strings, ausência de paralelismo.

\section{Melhor, Pior e Caso Médio}

\begin{center}
\begin{tabular}{lllll}
\textbf{Algoritmo} & \textbf{Melhor} & \textbf{Pior} & \textbf{Médio} & \textbf{Espaço} \\
\hline
Ingestão + tokenização & $\Theta(L)$ & $\Theta(L)$ & $\Theta(L)$ & $\Theta(L)$ \\
Busca linear (scoring) & $\Theta(N \cdot m)$ & $\Theta(N \cdot m)$ & $\Theta(N \cdot m)$ & $\Theta(N)$ \\
Busca indexada & $\Theta(m \log V)$ & $\Theta(m \log V + \sum n_t)$ & $\Theta(m \log V + \sum n_t)$ & $\Theta(L)$ \\
Insertion Sort & $\Theta(N)$ & $\Theta(N^2)$ & $\Theta(N^2)$ & $\Theta(1)$ \\
Merge Sort & $\Theta(N \log N)$ & $\Theta(N \log N)$ & $\Theta(N \log N)$ & $\Theta(N)$ \\
\end{tabular}
\end{center}

O ganho da busca indexada sobre a busca linear decorre de $\sum n_t \ll N \cdot m$: a maioria dos termos aparece em poucos documentos, seguindo aproximadamente uma distribuição de Zipf — hipótese sobre a distribuição dos dados que sustenta o caso médio.

\section{Recorrências}

\textbf{Merge Sort:} $T(N) = 2T(N/2) + \Theta(N)$, originada pelas duas chamadas recursivas sobre as metades da lista somadas ao custo linear da fusão. Pelo Teorema Mestre, caso 2 ($a=2$, $b=2$, $f(N)=\Theta(N)$), resolve-se em $T(N) = \Theta(N \log N)$.

\textbf{Busca binária no dicionário de termos:} $T(n) = T(n/2) + \Theta(1)$, originada pela bisseção do intervalo de busca a cada chamada. Pelo Teorema Mestre, caso 2 ($a=1$, $b=2$, $f(n)=\Theta(1)$), resolve-se em $T(n) = \Theta(\log n)$.

\section{Limitações e Ameaças à Validade}

\begin{itemize}
    \item O corpus original possui 167.053 notícias; por questões de tempo de execução e risco de estouro de memória, os experimentos utilizam duas amostras aleatórias fixas de 5.000 e 50.000 documentos, que não representam o corpus completo.
    \item O acervo cobre o período de janeiro de 2015 a setembro de 2017, o que limita a generalização dos resultados para conteúdo atual.
    \item O corpus provém de uma única fonte jornalística (Folha de S.Paulo), com viés editorial próprio e cobertura geográfica concentrada em Sudeste e Brasília.
    \item O BM25 é uma função de relevância puramente lexical, sujeita ao problema de \textit{vocabulary mismatch} (discutido na Seção~\ref{sec:rag}).
    \item Fatores de hardware, interpretador e ausência de paralelismo (Seção~10) introduzem variabilidade nas medições de tempo, mitigada por aquecimento e mediana de repetições.
\end{itemize}
