
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection for historical texts
collection = client.get_or_create_collection(
    name="historical_corpus",
    metadata={"description": "Diachronic multilingual corpus"}
)

# Initialize embedding model (open-source)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("Vector database initialized")
print(f"Collection: {collection.name}")
print(f"Documents: {collection.count()}")
