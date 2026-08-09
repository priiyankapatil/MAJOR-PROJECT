import os
import pickle, json, sys
os.environ.setdefault("NO_INTERACTIVE_FEEDBACK", "1")
from sentence_transformers import SentenceTransformer
import chromadb
from config import VECTOR_STORE, COLLECTION_NAME, EMBEDDING_MODEL
from step6_query_gate import query_gate

# Load embedder
print('Loading embedder...')
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Open ChromaDB
print('Opening ChromaDB...')
client = chromadb.PersistentClient(path=VECTOR_STORE)
try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception:
    collection = client.get_or_create_collection(COLLECTION_NAME)

# Load BM25 index and corpus
print('Loading BM25 and corpus...')
with open('knowledge_graph/bm25_index.pkl','rb') as f:
    bm25 = pickle.load(f)
with open('knowledge_graph/bm25_corpus.pkl','rb') as f:
    corpus = pickle.load(f)

queries = [
    "Arhar mein peela rog ka ilaj kya hai?",
    "My moong crop has maahu problem, what to do?",
    "Kali mitti mein gehun ke liye fertilizer kya use karein?",
    "Tana borer se jowar ko kaise bachayein?",
    "What is vermicompost?"
]

for q in queries:
    print('\n' + '='*40)
    print('Query:', q)
    try:
        res = query_gate(q, embedder, collection, bm25, corpus)
        print('\nResult keys:', list(res.keys()))
        # print short answer if present
        ans = res.get('answer') or (res.get('answer_data') and res['answer_data'].get('answer'))
        if ans:
            print('\nAnswer snippet:', ans[:400])
    except Exception as e:
        print('Query failed:', e)
