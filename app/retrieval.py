import json
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import ollama
import re
from difflib import get_close_matches
from qdrant_client.models import Filter, FieldCondition, MatchValue
from db import client, vector_store, storage_context, COLLECTION

# --- CONFIG ---
OLLAMA_BASE_URL = "http://host.docker.internal:11434"

# --- INIT ---
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text-v2-moe",
    base_url=OLLAMA_BASE_URL
)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)


JUNK_SECTIONS = {'references', 'citation', 'edited by', 'ethics statement', 
                 'acknowledgements', 'acknowledgments', 'funding', 'author contributions',
                 'conflict of interest', 'publisher\'s note'}

def resolve_paper_title(paper_title):
    """Find the closest matching stored paper title."""
    results = client.scroll(
        collection_name=COLLECTION,
        limit=100,
        with_payload=True,
    )
    stored_titles = list(set(r.payload.get("paper_title") for r in results[0]))
    
    matches = get_close_matches(paper_title, stored_titles, n=1, cutoff=0.4)
    if matches:
        return matches[0]
    
    # Fall back to substring match
    paper_lower = paper_title.lower()
    for title in stored_titles:
        if paper_lower in title.lower():
            return title
    return paper_title  # fallback to original if no match found


def retrieve(query, top_k=5, paper_title=None, paper_titles=None, score_threshold=0.5):
    # Always expand query into multiple sub-queries
    
    nodes = _do_retrieve(query=query, top_k=top_k, paper_title=paper_title, paper_titles=paper_titles)
    
    # Deduplicate and filter
    seen = set()
    unique = []
    for n in nodes:
        if n.node_id not in seen:
            seen.add(n.node_id)
            unique.append(n)
    unique = [n for n in unique 
              if n.metadata.get("section", "").lower() not in JUNK_SECTIONS]
    filtered = [n for n in unique if n.score >= score_threshold]
    return filtered if filtered else unique[:1], unique


def _do_retrieve(query, top_k=5, paper_title=None, paper_titles=None):
    """Raw retrieval without expansion."""
    kwargs = {}
    if paper_title:
        kwargs["vector_store_kwargs"] = {
            "qdrant_filters": Filter(
                must=[
                    FieldCondition(key="paper_title", match=MatchValue(value=paper_title))
                ]
            )
        }
    elif paper_titles:
        kwargs["vector_store_kwargs"] = {
            "qdrant_filters": Filter(
                should=[
                    FieldCondition(key="paper_title", match=MatchValue(value=t))
                    for t in paper_titles
                ]
            )
        }
    retriever = index.as_retriever(similarity_top_k=top_k, **kwargs)
    return retriever.retrieve(query)


def deduplicate_nodes(nodes, similarity_threshold=0.8):
    """Remove truly redundant chunks based on text overlap."""
    result = []
    for node in nodes:
        is_duplicate = False
        for kept in result:
            # Simple overlap check using set of words
            words_a = set(node.text.lower().split())
            words_b = set(kept.text.lower().split())
            overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
            if overlap > similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            result.append(node)
    return result


def enrich_with_references(nodes):
    """Pull linked figures and tables for retrieved text chunks."""
    enriched = []
    for node in nodes:
        result = {
            "text": node.text,
            "metadata": node.metadata,
            "score": node.score,
            "linked": []
        }

        all_refs = (
            node.metadata.get("referenced_figures", []) +
            node.metadata.get("referenced_tables", []) +
            node.metadata.get("referenced_equations", [])
        )

        for ref_id in all_refs:
            if ref_id.split("_")[0] == "supp":  
                ref_type = ref_id.split("_")[0] + "_" + ref_id.split("_")[1]  # e.g. "supp_figure"
            else:
                ref_type = ref_id.split("_")[0]
            try:
                linked = client.scroll(
                    collection_name=COLLECTION,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key=f"{ref_type}_id",
                                match=MatchValue(value=ref_id)
                            ),
                            FieldCondition(
                                key="paper_title",
                                match=MatchValue(value=node.metadata.get("paper_title", ""))
                            ),
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                )
                if linked[0]:
                    result["linked"].append(linked[0][0].payload)
            except Exception as e:
                print(f"Reference lookup failed for {ref_id}: {e}")

        enriched.append(result)
    return enriched


def get_figure_image_on_demand(pdf_path, page_no, bbox_l, bbox_t, bbox_r, bbox_b):
    """Re-render a specific page and crop the figure bbox."""
    from PIL import Image
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    page = doc[page_no - 1]  # 0-indexed

    # Render page at high resolution
    mat = fitz.Matrix(2.0, 2.0)  # 2x scale
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Scale bbox coordinates to match rendered resolution
    page_width = page.rect.width
    page_height = page.rect.height
    scale_x = pix.width / page_width
    scale_y = pix.height / page_height

    # Crop to figure bbox
    cropped = img.crop((
        bbox_l * scale_x,
        (page_height - bbox_t) * scale_y,  # docling uses inverted y-axis
        bbox_r * scale_x,
        (page_height - bbox_b) * scale_y,
    ))

    return cropped


def keyword_retrieve(query, paper_title=None, paper_titles=None, limit=5):
    """Keyword-based retrieval using text matching."""
    # Extract meaningful keywords — skip common words
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'which',
                 'does', 'do', 'did', 'can', 'could', 'would', 'should', 'and', 
                 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 
                 'each', 'other', 'seperately', 'use',
                 'describe', 'explain', 'compare', 'contrast', 'similar', 'different'}
    keywords = list(dict.fromkeys(
                w for w in re.findall(r'\b[A-Za-z][A-Za-z0-9\-]+\b', query)
                if w.lower() not in stopwords and len(w) > 2
                ))  # preserve order and remove duplicates
    if not keywords:
        return []

    # Get all text chunks for the relevant papers
    filter_conditions = [FieldCondition(key="type", match=MatchValue(value="text"))]
    if paper_title:
        filter_conditions.append(FieldCondition(key="paper_title", match=MatchValue(value=paper_title)))

    results = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(must=filter_conditions),
        limit=500,
        with_payload=True,
    )

    # Score by keyword hits
    scored = []
    for r in results[0]:
        # Check paper_titles filter in Python
        if paper_titles and r.payload.get("paper_title") not in paper_titles:
            continue
        
        text = json.loads(r.payload.get("_node_content", "{}")).get("text", "").lower()
        section = r.payload.get("section", "").lower()
        
        score = 0
        for kw in keywords:
            kw_lower = kw.lower()
            # Higher weight for exact model/method names (capitalized)
            if kw[0].isupper():
                score += text.count(kw_lower) * 2
                score += section.count(kw_lower) * 3  # section name match is strong signal
            else:
                score += text.count(kw_lower)
        
        if score > 0:
            scored.append((score, r.payload))
    
    # Sort by score and return top results
    scored.sort(key=lambda x: x[0], reverse=True)
    return [payload for _, payload in scored[:limit]]


def hybrid_retrieve(query, top_k=5, paper_title=None, paper_titles=None, score_threshold=0.5):
    """Combine semantic and keyword retrieval."""
    # Semantic retrieval
    semantic_nodes, nodes = retrieve(query, top_k=top_k, paper_title=paper_title, 
                              paper_titles=paper_titles, score_threshold=score_threshold)
    
    # Keyword retrieval
    keyword_payloads = keyword_retrieve(query, paper_title=paper_title, 
                                        paper_titles=paper_titles, limit=3)
    
    # Convert keyword results to node-like dicts for enrichment
    keyword_nodes = []
    for payload in keyword_payloads:
        # Check not already in semantic results
        semantic_texts = {n.text for n in semantic_nodes}
        text = json.loads(payload.get("_node_content", "{}")).get("text", "")
        if text and text not in semantic_texts:
            keyword_nodes.append({
                "text": text,
                "metadata": {k: v for k, v in payload.items() 
                            if k not in ("_node_content", "_node_type")},
                "score": 0.0,  # keyword match has no similarity score
                "linked": [],
                "source": "keyword"
            })
    return semantic_nodes, keyword_nodes, nodes


def handle_direct_query(query, paper_title=None):
    """Detect if user is asking about a specific figure, table, or equation and fetch directly."""
    
    patterns = {
        "figure": r'Fig(?:ure)?\.?\s*(\d+)',
        "table": r'Table\.?\s*(\d+)',
        "equation": r'(?:Equation|Eq\.?)\s*\(?(\d+)\)?',
    }
    if paper_title is None:
        return None, None
    
    for ref_type, pattern in patterns.items():
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            ref_id = f"{ref_type}_{match.group(1)}"
            id_key = f"{ref_type}_id"

            results = client.scroll(
                collection_name=COLLECTION,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key=id_key, match=MatchValue(value=ref_id)),
                        FieldCondition(key="paper_title", match=MatchValue(value=paper_title)),
                    ]
                ),
                limit=1,
                with_payload=True,
            )

            if not results[0]:
                return None, None

            payload = results[0][0].payload

            # ── Fetch referring text chunks ───────────────────────────
            ref_chunks = client.scroll(
                collection_name=COLLECTION,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key=f"referenced_{ref_type}s",
                            match=MatchValue(value=ref_id)
                        ),
                        FieldCondition(
                            key="paper_title",
                            match=MatchValue(value=paper_title)
                        ),
                    ]
                ),
                limit=5,
                with_payload=True,
            )

            referring_texts = [p.payload for p in ref_chunks[0]] if ref_chunks[0] else []

            # ── For figures: re-render image from PDF ─────────────────
            pil_image = None
            if ref_type == "figure":
                try:
                    pil_image = get_figure_image_on_demand(
                        pdf_path=payload.get("source_pdf"),
                        page_no=payload.get("page"),
                        bbox_l=payload.get("bbox_l"),
                        bbox_t=payload.get("bbox_t"),
                        bbox_r=payload.get("bbox_r"),
                        bbox_b=payload.get("bbox_b"),
                    )
                except Exception as e:
                    print(f"Could not re-render figure: {e}")

            return {
                "payload": payload,
                "ref_type": ref_type,
                "ref_id": ref_id,
                "referring_texts": referring_texts,
                "pil_image": pil_image,
            }, ref_type
    
    return None, None

