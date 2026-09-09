import pandas as pd
import re

def limpar_texto(texto):

    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto

def gerar_chunks(texto, tamanho_chunk=100):
    
    palavras = texto.split()
    return [" ".join(palavras[i:i + tamanho_chunk]) for i in range(0, len(palavras), tamanho_chunk)]

def carregar_e_processar_corpus(caminho_csv):
    y
    print("Iniciando ingestão do CSV...")
    df = pd.read_csv(caminho_csv)
    
    print("Normalizando textos...")
    df['texto_limpo'] = df['text'].apply(limpar_texto)
    df['titulo_limpo'] = df['title'].apply(limpar_texto) 
    
    print("Gerando chunks...")
    corpus_processado = []
    
    for idx, row in df.iterrows():
        chunks = gerar_chunks(row['texto_limpo'])
        for i, chunk in enumerate(chunks):
            corpus_processado.append({
                'doc_id': f"{idx}_chunk_{i}",
                'titulo': row['titulo_limpo'],
                'texto': chunk
            })
            
    df_final = pd.DataFrame(corpus_processado)
    df_final.to_csv('corpus_normalizado.csv', index=False)
    print(f"Processamento concluído. {len(df_final)} chunks gerados.")
    return df_final

if __name__ == "__main__":
    caminho = "articles.csv"
    carregar_e_processar_corpus(caminho)