import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient('vector_store')
collection = client.get_collection('agriculture_knowledge')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Test query that should match Crop-Management_1.pdf
query = 'crop management techniques'
emb = embedder.encode([query]).tolist()

results = collection.query(query_embeddings=emb, n_results=10, include=['documents', 'metadatas', 'distances'])

print('Top 10 results for: "crop management techniques"')
print('='*70)
for i, (doc_id, doc, meta, dist) in enumerate(zip(
    results['ids'][0],
    results['documents'][0],
    results['metadatas'][0],
    results['distances'][0]
), 1):
    sim = round(1 - dist, 4)
    print(f'{i}. [{sim}] {meta["source_file"]}')
    print(f'   {doc[:80]}...')
    print()
