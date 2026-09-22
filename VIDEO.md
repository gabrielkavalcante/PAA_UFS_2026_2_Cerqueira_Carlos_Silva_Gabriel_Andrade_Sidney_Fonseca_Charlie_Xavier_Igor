# Vídeo da Atividade — Projeto e Análise de Algoritmos (2026.2)

> **Atividade 1 (AV1):** Corretude, eficiência e recuperação de contexto para IA generativa  
> **Universidade Federal de Sergipe (UFS)** — Departamento de Computação  
> **Tema 6:** Notícias de Fontes Públicas (Folha de S.Paulo / UOL)

---

## Informações do Vídeo

* **URL do Vídeo:** `[URL PENDENTE DE GRAVAÇÃO - INSERIR LINK DO YOUTUBE/DRIVE PÚBLICO]`
* **Data da Gravação:** `[DD/MM/2026]`
* **Duração Total:** `[MM:SS]` *(máximo de 10 minutos, conforme edital)*
* **Plataforma de Hospedagem:** `[YouTube (Não listado) / Google Drive (Acesso público)]`

---

## Participantes e Divisão das Falas

Conforme estabelecido na **Seção 11 do edital**, todos os integrantes participam ativamente da apresentação, abordando suas respectivas contribuições técnicas e teóricas:

| Integrante | Papel no Projeto | Tópico Apresentado no Vídeo | Tempo Estimado |
| :--- | :--- | :--- | :---: |
| **Gabriel Cavalcante Silva** | Engenharia de Dados e Infraestrutura | Introdução, definição do tema/corpus (Folha de S.Paulo), arquitetura do pipeline e demonstração do protótipo | ~2:00 min |
| **Carlos Adriano Gama Cerqueira** | Engenharia de Software: Algoritmos | Implementação dos algoritmos: busca linear (C1), índice invertido com tabela hash (C2/C3), Insertion Sort e Merge Sort | ~2:00 min |
| **Sidney Matheus Santos de Andrade** | Ciência da Computação: Teoria | Fundamentação teórica: prova formal de corretude (invariante de laço e indução), análise no modelo RAM e recorrências | ~2:00 min |
| **Igor Rafael Silva Xavier** | Análise de Desempenho: Testes | Metodologia experimental de desempenho, discussão dos tempos de consulta, consumo de memória e análise dos gráficos | ~2:00 min |
| **Charlie Rodrigues Fonseca** | Análise Qualitativa e Documentação | Avaliação de qualidade da recuperação (Precision@5 e *vocabulary mismatch*), distribuição de tokens (sem *chunking*), impacto em RAG e uso de IA | ~2:00 min |

---

## Roteiro e Estrutura da Apresentação (Checklist do Edital)

O vídeo contempla integralmente os 10 tópicos obrigatórios exigidos pela especificação:

1. **Abertura e Contextualização:** Identificação da equipe, objetivo do projeto, tema e corpus de notícias públicas.
2. **Problema de Recuperação de Contexto:** Formalização da tarefa de busca lexical sobre $N$ documentos e retorno dos $k=5$ mais relevantes para um LLM.
3. **Algoritmos Implementados:** Apresentação comparativa das configurações C1 (Busca linear + Insertion), C2 (Índice hash + Insertion), C3 (Índice hash + Merge Sort) e C4 (Baseline externo: `rank_bm25`).
4. **Demonstração Prática:** Execução ao vivo do protótipo recebendo consultas reais e retornando os documentos ranqueados.
5. **Justificativa Formal de Corretude:** Explicação sucinta do invariante de laço do Insertion Sort e do passo indutivo do Merge Sort sob a ordem total definida.
6. **Análise Assintótica:** Comparativo RAM entre $\Theta(n^2)$ e $\Theta(n \log n)$, com a resolução da recorrência $T(n) = 2T(n/2) + \Theta(n)$ via Teorema Mestre.
7. **Resultados de Desempenho:** Discussão dos dados experimentais (300 execuções), tempos medianos e números reais de comparações.
8. **Avaliação de Qualidade e Tokens:** Apresentação da Precision@5 (0,76 a 1,00), do *vocabulary mismatch* e da análise empírica de tokens (Q1, Q3, P95) embasando a dispensa de *chunking*.
9. **Implicações para RAG e IA Generativa:** Dimensionamento do prompt, previsibilidade de contexto, limitações da busca puramente lexical e trade-offs de engenharia.
10. **Declaração de Uso de IA e Conclusão:** Transparência sobre o auxílio de LLMs no projeto com supervisão humana estrita e encerramento.

