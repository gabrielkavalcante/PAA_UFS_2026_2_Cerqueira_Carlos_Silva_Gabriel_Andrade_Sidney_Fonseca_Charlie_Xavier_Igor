# PAA - Projeto e Análise de Algoritmos (2026.2)
Tema 6: Notícias de Fontes Públicas

## Equipe
* Carlos Adriano Gama Cerqueira
* Gabriel Cavalcante Silva
* Sidney Matheus Santos de Andrade
* Charlie Rodrigues Fonseca
* Igor Rafael Silva Xavier

---

## Sobre o Corpus e Acesso aos Dados
O projeto utiliza o Dataset de Notícias Folha/UOL extraído da plataforma Kaggle. Como os arquivos de dados brutos possuem grande volume e não são versionados diretamente no Git conforme as diretrizes do edital, o arquivo `articles.csv` deve ser obtido através do link externo da equipe:

* **Link para download do Corpus Normalizado:** [Acessar Corpus no Google Drive](https://drive.google.com/file/d/1KfPinMM6bRGGy5FNvC_DWqFoAJJpXYBu/view?usp=sharing)

---

## Estrutura do Repositório
* **`preprocessamento.py`**: Script em Python responsável pela ingestão, limpeza, normalização e segmentação (*chunking*) dos textos do corpus.
* **`relatorio_base.tex`**: Esqueleto oficial do relatório técnico em LaTeX contendo as 19 seções obrigatórias.
* **`.gitignore`**: Regras de exclusão para impedir o envio acidental de arquivos pesados (*CSVs* e artefatos de compilação) para o repositório remoto.

---

## Instruções de Execução

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/gabrielkavalcante/PAA_UFS_2026_2_Cerqueira_Carlos_Silva_Gabriel_Andrade_Sidney_Fonseca_Charlie_Xavier_Igor.git
   cd PAA_UFS_2026_2_Cerqueira_Carlos_Silva_Gabriel_Andrade_Sidney_Fonseca_Charlie_Xavier_Igor

---

### Como atualizar no seu projeto:
1. Abra o arquivo `README.md` no seu VS Code, apague tudo o que tem nele e cole o texto acima.
2. Salve o arquivo (`Ctrl + S`).
3. No terminal do VS Code, envie a alteração final para o GitHub com os comandos:
   ```bash
   git add README.md
   git commit -m "Adiciona link oficial do corpus normalizado no README"
   git push