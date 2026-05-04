import json
from datetime import datetime
from pathlib import Path
from llama_index.core import StorageContext, VectorStoreIndex
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from embedding import process_research_paper
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from tqdm import tqdm


COLLECTION = "research_papers"
PDF_DIR = Path("/app/documents") 
QDRANT_PATH = "/app/qdrant_db"
REGISTRY_PATH = "/app/qdrant_db/registry.json"
BACKUP_PATH = "/app/qdrant_db/nodes_backup.json"
OLLAMA_BASE_URL = "http://host.docker.internal:11434"

client = QdrantClient(path=QDRANT_PATH)
vector_store = QdrantVectorStore(collection_name=COLLECTION, client=client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Initialize Nomic Embeddings via Ollama
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text-v2-moe",
    base_url=OLLAMA_BASE_URL
)


def save_nodes(nodes, path="nodes_backup.json"):
    data = []
    for node in nodes:
        data.append({
            "text": node.text,
            "metadata": node.metadata,
            "id": node.id_,
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"💾 Saved {len(nodes)} nodes to {path}")

def load_nodes(path="nodes_backup.json"):
    with open(path) as f:
        data = json.load(f)
    nodes = []
    for d in data:
        node = TextNode(text=d["text"], metadata=d["metadata"], id_=d["id"])
        nodes.append(node)
    print(f"📂 Loaded {len(nodes)} nodes from {path}")
    return nodes

def load_registry():
    if Path(REGISTRY_PATH).exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}

def save_registry(registry):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

def is_already_ingested(pdf_name):
    return pdf_name in load_registry()

def register_pdf(pdf_name, paper_title, node_count):
    registry = load_registry()
    registry[pdf_name] = {
        "paper_title": paper_title,
        "node_count": node_count,
        "ingested_at": str(datetime.now()),
    }
    save_registry(registry)

def delete_pdf(pdf_name):
    registry = load_registry()
    
    if pdf_name not in registry:
        print(f"❌ {pdf_name} not found in registry")
        return
    
    paper_title = registry[pdf_name]["paper_title"]
    
    # Delete from Qdrant
    client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="paper_title",
                    match=MatchValue(value=paper_title)
                )
            ]
        )
    )
    print(f"🗑️ Deleted vectors for {paper_title}")
    
    # Remove from backup
    if Path(BACKUP_PATH).exists():
        existing = load_nodes(BACKUP_PATH)
        filtered = [n for n in existing if n.metadata.get("paper_title") != paper_title]
        save_nodes(filtered, BACKUP_PATH)
    
    # Remove from registry
    del registry[pdf_name]
    save_registry(registry)
    print(f"✅ Removed {pdf_name} from registry")

if __name__ == "__main__":

    # To delete a specific PDF, uncomment and run:
    #delete_pdf("Xu et al. - 2021 - Progression of sleep disturbances in Parkinson’s d.pdf")

    print("🚀 Starting Ingestion...")
    pdf_paths = sorted(Path(PDF_DIR).glob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDFs")

    all_new_nodes = []
    for path in tqdm(pdf_paths, desc="PDFs"):
        if is_already_ingested(path.name):
            print(f"⏭️ Skipping {path.name} — already ingested")
            continue

        print(f"\nProcessing {path.name}...")
        doc_nodes = process_research_paper(path)
        print(f"  → {len(doc_nodes)} nodes")
        all_new_nodes.extend(doc_nodes)

        # Register immediately after processing
        # Get paper title from first node
        paper_title = doc_nodes[0].metadata.get("paper_title", "Unknown") if doc_nodes else "Unknown"
        register_pdf(path.name, paper_title, len(doc_nodes))

    if all_new_nodes:
        # Update backup with new nodes
        if Path(BACKUP_PATH).exists():
            existing = load_nodes(BACKUP_PATH)
            all_nodes = existing + all_new_nodes
        else:
            all_nodes = all_new_nodes
        save_nodes(all_nodes, BACKUP_PATH)

        print(f"\nIndexing {len(all_new_nodes)} new nodes into Qdrant...")
        index = VectorStoreIndex(
            all_new_nodes,  # only index new nodes
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )
    else:
        print("\nNo new PDFs to process — loading existing index...")
        # Load existing index from Qdrant
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
        )

    print("✅ Done!")