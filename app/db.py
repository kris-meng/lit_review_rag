from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext

QDRANT_PATH = "/app/qdrant_db"
COLLECTION = "research_papers"

client = QdrantClient(path=QDRANT_PATH)
vector_store = QdrantVectorStore(collection_name=COLLECTION, client=client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)