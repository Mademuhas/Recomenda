from genai.credentials import Credentials
from genai.model import Model
from genai.schemas import GenerateParams
import pandas as pd
import os
from dotenv import load_dotenv
# import tensorflow_hub as hub
import json
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sentence_transformers import SentenceTransformer

def get_dataframe (arquivo):
    df = pd.read_csv(arquivo, delimiter=';',encoding='utf-8')
    return df


def get_the_model ():
    ### Lendo as credenciais para utilizar o bam
    load_dotenv()
    api_key = "pak-vlVlI1mqIZEPHHKA-GblooCrmJYVdfguI7zryvTTgvg"
    api_url = "https://bam-api.res.ibm.com/v1"
    ### Chamando o modelo para ser consumido
    creds = Credentials(api_key, api_url)
    params = GenerateParams(decoding_method="greedy", max_new_tokens=1500,stop_sequences=["""}\n\n"""])
    llama_70b_chat = Model("meta-llama/llama-2-70b-chat", params=params, credentials=creds)

    return llama_70b_chat

### Função padrão para gerar um prompt
def make_prompt (entrada, instruction):
    prompt = f"""
    {instruction}
    
Entrada:
{entrada}
    
Saida:
[/INST]
"""
    
    return prompt

# Função em que o usuário passa o pedido e ela retorna o Json referente com as instruções extraidas do llama
def get_the_json(entrada):

    llama_70b_chat = get_the_model()
    instruction = """
    <s>[INST] <<SYS>>
    Você trabalha em uma loja de vinhos e é responsável por atender os clientes. O cliente te fornecerá detalhes do que ele espera
    em um vinho, você deve extrair as principais informações no pedido do cliente. Por exemplo, se o cliente informar, 
    você deve identificar o tipo de vinho, as comidas que harmonizam e a descrição do vinho.
    Caso não for possível identificar alguma dessas informações, basta preencher em branco. Sua saída deverá ser em formato JSON,
    atente-se as vírgulas, espaços e chaves. A seguir estão alguns exemplos:

    Entrada:

    Gostaria de um vinho seco bom para tomar comendo peixe

    Saída:

    {
        "Tipo" : "Seco",
        "Harmoniza" : "Peixe",
        "Descriçao" : ""
    }

    Entrada:

    Quero um vinho suave feito de maneira organica e em ambiente frio, para comer com carne de porco e cogumelos


    Saída:

    {
        "Tipo" : "Suave",
        "Harmoniza" : "Carne de Porco e Cogumelos",
        "Descriçao" : "feito de maneira organica e em ambiente frio"
    }

    Entrada:

    Oi, quais vinhos são doce, perfeitos para dias de calor e que combinem muito bem com aperetivos?
    Saída:

    {
        "Tipo" : "doce",
        "Harmoniza" : "Aperitivos",
        "Descriçao" : "perfeitos para dias de calor"
    }



    <</SYS>>

    """

    teste_prompt = make_prompt(entrada, instruction)
    result = llama_70b_chat.generate([teste_prompt])
    tj = json.loads(result[0].generated_text)
    return tj

      

def get_vinho(obj):
    df = pd.read_csv('novos.csv')
    df = df.drop(columns='Unnamed: 0')
    lista_harmo = df['harmoniza']
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    lista_embedding = []
    for i in range (len(lista_harmo)):
         embedding_pm = model.encode (f'{lista_harmo[i]}')
         lista_embedding.append(embedding_pm)

    X = np.array(lista_embedding)
    print (X.shape)
    emb_pnr = model.encode (f'{obj}')

    Y = np.array(emb_pnr)
    Y = Y.reshape(1, -1)
    print (Y.shape)
    nn = NearestNeighbors(n_neighbors=3)
    nn.fit(X)
   
    neighbors = nn.kneighbors(Y, return_distance=False)
    knn = neighbors
    idx = knn[0][0]

    return (df.iloc[idx].to_json(orient = "columns"))
