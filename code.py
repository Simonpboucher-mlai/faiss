from fastapi import FastAPI, HTTPException, Header, Depends, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import requests
import json
import os
import faiss
import tiktoken  # Importer tiktoken pour compter les tokens

app = FastAPI()

# Configuration du middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mettez ici les origines que vous autorisez
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Récupérer la clé API depuis les variables d'environnement
API_KEY = "sk-proj"
if not API_KEY:
    raise ValueError("La clé API OpenAI n'est pas définie dans les variables d'environnement.")

MODEL = "gpt-4o-mini"

# Token d'accès (à configurer selon vos besoins)
ACCESS_TOKEN = "m-lai-CaNYFR1GolGVp7uY8sQ51cSU35X3kB7lGx"

# Vérification du token d'accès
def verify_token(authorization: str = Header(...)):
    if authorization != f"Bearer {ACCESS_TOKEN}":
        raise HTTPException(status_code=401, detail="Token d'accès invalide.")

# Fonction pour générer un embedding avec OpenAI
def generate_openai_embedding(text, model):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "model": model
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # Vérifier les erreurs HTTP
    return response.json()["data"][0]["embedding"]

# Initialiser l'encodeur de tokens pour OpenAI
tokenizer = tiktoken.get_encoding("cl100k_base")  # Utiliser l'encodeur correspondant au modèle GPT

# Fonction pour calculer le nombre de tokens dans un chunk
def count_tokens(chunk):
    return len(tokenizer.encode(chunk))

# Global dictionary to store indices and chunks per company_id
company_indices = {}

# Fonction pour rechercher les chunks similaires en respectant la limite de 8000 tokens
def search_similar_chunks(query_embedding, company_id, token_limit=8000):
    global company_indices

    try:
        # Vérifier si l'index pour le company_id existe déjà
        if company_id not in company_indices:
            # Charger les embeddings et les chunks
            embeddings = np.load(f"./files/{company_id}/embeddings.npy").astype('float32')
            with open(f"./files/{company_id}/chunks.json", "r") as f:
                chunks = json.load(f)

            # Normaliser les embeddings pour la similarité cosinus
            faiss.normalize_L2(embeddings)

            # Construire l'index FAISS
            index = faiss.IndexFlatIP(embeddings.shape[1])  # Utiliser le produit scalaire pour la similarité cosinus
            index.add(embeddings)

            # Stocker l'index et les chunks
            company_indices[company_id] = {'index': index, 'chunks': chunks}
        else:
            index = company_indices[company_id]['index']
            chunks = company_indices[company_id]['chunks']

        # Normaliser l'embedding de la requête
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding.reshape(1, -1))

        # Recherche initiale des résultats (récupérer tous les chunks)
        top_k = len(chunks)
        distances, indices_results = index.search(query_embedding.reshape(1, -1), top_k)

        # Récupérer les chunks similaires
        similar_chunks = [(distances[0][i], chunks[indices_results[0][i]]) for i in range(len(indices_results[0]))]

        # Accumuler les chunks jusqu'à atteindre la limite de 8000 tokens
        total_tokens = 0
        selected_chunks = []
        for _, chunk in similar_chunks:
            chunk_tokens = count_tokens(chunk)
            if total_tokens + chunk_tokens > token_limit:
                break
            selected_chunks.append(chunk)
            total_tokens += chunk_tokens

        return selected_chunks

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search: {str(e)}")


# Modèles Pydantic pour valider la requête et l'historique
class Message(BaseModel):
    role: str  # "user" ou "assistant"
    content: str

class UserQuery(BaseModel):
    company_id: str
    question: str
    embedding_model: str

@app.post("/query")
async def query_bot(user_query: UserQuery, authorization: str = Depends(verify_token)):
    try:
        # Générer l'embedding pour la question de l'utilisateur
        query_embedding = generate_openai_embedding(user_query.question, user_query.embedding_model)

        # Rechercher les chunks similaires en respectant la limite de 8000 tokens
        similar_chunks = search_similar_chunks(query_embedding, user_query.company_id, token_limit=8000)

        return Response(
            json.dumps(similar_chunks),
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Querybot: {str(e)}")


@app.post("/uploadfile/{company_id}")
async def upload_file(file: UploadFile, company_id: str, authorization: str = Depends(verify_token)):
    try:
        file_path = f"./files/{company_id}/{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
            return {"message": "File saved successfully"}
    except Exception as e:
        return {"message": e.args}
