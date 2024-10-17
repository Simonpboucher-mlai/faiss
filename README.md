# faiss

import numpy as np
import json
import faiss
from openai import OpenAI
import os

# Assurez-vous de remplacer ceci par votre véritable clé API OpenAI
api_key = ""
client = OpenAI(api_key=api_key)

# Charger les embeddings et les chunks
embeddings = np.load('embedding.npy')
with open('chunk.json', 'r') as f:
    chunks = json.load(f)

# Créer l'index FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Index pour la similarité cosinus
index.add(embeddings)

def get_embedding(text):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        print(f"Erreur lors de l'obtention de l'embedding: {e}")
        return None

def create_context(question, max_len=8000, k=5):
    q_embedding = get_embedding(question)
    if q_embedding is None:
        return ""

    # Recherche des k plus proches voisins
    distances, indices = index.search(q_embedding.reshape(1, -1), k)

    returns = []
    cur_len = 0

    for i in indices[0]:
        cur_len += len(chunks[i]['text'].split()) + 4
        if cur_len > max_len:
            break
        returns.append(chunks[i]["text"])

    return "\n\n###\n\n".join(returns)

def answer_question(question, max_len=8000, max_tokens=9000):
    context = create_context(question, max_len=max_len)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for the company Ouellet, which Provide sustainable solutions to the HVAC (heating, ventilation, air conditioning) industry by offering systems designed to maximize residential comfort. This commitment is made possible by both the undertaking of our employees and the contribution of our partners in innovation. You help user find the best product from the large evantail of choice. If you dont have the answer, try to anmswer with common sense and dont said the context doesnt provide the indo. Never say that. An always give detail of your statement"},
                {"role": "user", "content": f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"}
            ],
            max_tokens=max_tokens,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Une exception s'est produite lors de la génération de la réponse: {e}")
        return ""

def chatbot(question):
    print(f"\033[1mQuestion :\033[0m {question}")

    print("\033[1mRéponse :\033[0m")
    answer = answer_question(question=question)
    print(answer)

    return {'faiss': answer}


    # Define your queries
queries = [
"Donne info sur OWC-R"
]

# Loop through queries and get response from chatbot
for query in queries:
    response = chatbot(query)
