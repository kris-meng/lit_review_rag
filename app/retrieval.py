import base64
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
import io

# --- CONFIG ---
QDRANT_PATH = "/app/qdrant_db"
COLLECTION = "research_papers"
OLLAMA_BASE_URL = "http://host.docker.internal:11434"

# --- INIT ---
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text-v2-moe",
    base_url=OLLAMA_BASE_URL
)

client = QdrantClient(path=QDRANT_PATH)
vector_store = QdrantVectorStore(collection_name=COLLECTION, client=client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

SYSTEM_PROMPT = """You are a research assistant helping with academic literature review.
You are given retrieved excerpts from research papers.

Rules:
- Always cite which paper and section your answer comes from
- If multiple papers say different things, highlight the contradiction
- If the retrieved context doesn't answer the question, say so explicitly
- For formulas, refer to them by their equation number
- Never make up information not present in the retrieved context"""

GLOBAL_TRIGGERS = [
    "other papers", "any papers", "compare", "across papers",
    "similar to", "different from", "address", "limitation",
    "weakness", "gap", "related work", "literature"
]

def detect_scope(query, paper_title=None):
    """Determine if query needs local or global search."""
    if paper_title is None:
        return "global"
    query_lower = query.lower()
    if any(trigger in query_lower for trigger in GLOBAL_TRIGGERS):
        return "global"
    return "local"

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
    expanded = expand_query(query)
    
    all_nodes = []
    for q in expanded:
        nodes = _do_retrieve(q, top_k=top_k, paper_title=paper_title, paper_titles=paper_titles)
        all_nodes.extend(nodes)
    
    # Deduplicate and filter
    seen = set()
    unique = []
    for n in all_nodes:
        if n.node_id not in seen:
            seen.add(n.node_id)
            unique.append(n)
    
    filtered = [n for n in unique if n.score >= score_threshold]
    return filtered if filtered else unique[:1]


def expand_query(query):
    """Always expand query into 3 variants for better retrieval coverage."""
    ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
    res = ollama_client.chat(model='qwen2.5:7b', messages=[{
        'role': 'user',
        'content': f"""Generate 3 different search queries to retrieve relevant academic paper chunks for this question:
{query}

Output only the queries, one per line, no numbering, no explanation."""
    }])
    queries = [q.strip() for q in res['message']['content'].strip().split('\n') if q.strip()]
    print(f"   Expanded queries: {queries}")
    return [query] + queries[:2]  # original + 2 expansions


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
            node.metadata.get("referenced_tables", [])
        )

        for ref_id in all_refs:
            ref_type = ref_id.split("_")[0]  # "figure" or "table"
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


def build_context(enriched_nodes):
    """Pack retrieved chunks into a prompt context string."""
    parts = []
    for node in enriched_nodes:
        meta = node["metadata"]
        header = (
            f"[{meta.get('type', '?').upper()} | "
            f"{meta.get('paper_title', '?')} | "
            f"Section: {meta.get('section', '?')} | "
            f"Page: {meta.get('page', '?')}]"
        )
        parts.append(f"{header}\n{node['text']}")

        # Append any linked figures/tables
        for linked in node["linked"]:
            linked_header = (
                f"  [LINKED {linked.get('type', '?').upper()} | "
                f"{linked.get('figure_id') or linked.get('table_id', '?')}]"
            )
            parts.append(f"{linked_header}\n  {linked.get('text', '')[:300]}...")

    return "\n\n---\n\n".join(parts)


def generate_answer(query, context, history=[]):
    """Call LLM with retrieved context."""
    ollama_client = ollama.Client(host=OLLAMA_BASE_URL)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({
        "role": "user",
        "content": f"Context from papers:\n{context}\n\nQuestion: {query}"
    })

    res = ollama_client.chat(model='qwen2.5:7b', messages=messages)
    return res['message']['content']


def check_relevancy(query, context, answer):
    """Check if answer is grounded in context."""
    ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
    res = ollama_client.chat(model='qwen2.5:7b', messages=[{
        'role': 'user',
        'content': f"""Given this question:
{query}

And this context:
{context[:2000]}

And this answer:
{answer}

Rate on two dimensions:
1. GROUNDEDNESS (0-1): Is the answer fully supported by the context?
2. RELEVANCE (0-1): Does the answer actually address the question?

Respond in JSON only, no explanation:
{{"groundedness": 0.0, "relevance": 0.0}}"""
    }])

    try:
        content = res['message']['content'].strip()
        content = re.sub(r'```json|```', '', content).strip()
        scores = json.loads(content)
        values = list(scores.values())
        return {
            "groundedness": float(values[0]),
            "relevance": float(values[1]),
        }
    except Exception as e:
        print(f"Relevancy parse failed: {e} — raw: {res['message']['content'][:100]}")
        return {"groundedness": 0.0, "relevance": 0.0}


def contextualize_query(query, history):
    """Rewrite query using conversation history for better retrieval."""
    if not history:
        return query

    ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
    res = ollama_client.chat(model='qwen2.5:7b', messages=[{
        'role': 'user',
        'content': f"""Given this conversation history:
{json.dumps(history[-4:], indent=2)}

And this new question: {query}

Rewrite the question as a standalone query that includes all necessary context.
Output only the rewritten query, nothing else."""
    }])
    return res['message']['content'].strip()


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


def chat(query, history=[], paper_title=None, paper_titles=None, top_k=5):
    contextualized = contextualize_query(query, history)
    print(f"\n🔍 Searching for: {contextualized}")

    if paper_title:
        paper_title = resolve_paper_title(paper_title)
        print(f"   Resolved paper title: {paper_title}")

    if paper_titles:
        paper_titles = [resolve_paper_title(t) for t in paper_titles]
        print(f"   Resolved paper titles: {paper_titles}")

    # ── Direct lookup for figure/table/equation queries ───────────────
    direct_result, ref_type = handle_direct_query(contextualized, paper_title)
    if direct_result:
        payload = direct_result["payload"]
        referring_texts = direct_result["referring_texts"]
        pil_image = direct_result.get("pil_image")

        print(f"   Direct {ref_type} lookup — bypassing vector search")
        print(f"   Found in: {payload.get('paper_title')}")
        print(f"   {len(referring_texts)} referring text chunks found")

        # Build context from payload + referring chunks
        context_parts = [
            f"[{ref_type.upper()} | {payload.get('paper_title')} | "
            f"Section: {payload.get('section')} | Page: {payload.get('page')}]\n"
            f"{json.loads(payload.get('_node_content', '')).get('text', '')}"
        ]
        for chunk in referring_texts:
            context_parts.append(
                f"[REFERRING TEXT | Section: {chunk.get('section')} | Page: {chunk.get('page')}]\n"
                f"{json.loads(chunk.get('_node_content', '')).get('text', '')}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # For figures: re-query VLM with specific question
        if ref_type == "figure" and pil_image:
            print("   Re-querying VLM with specific question...")
            vlm_answer = query_figure_with_question(pil_image, query, payload.get("caption", ""))
            context += f"\n\n---\n\n[VLM ANALYSIS FOR QUERY]\n{vlm_answer}"

        answer = generate_answer(query, context, history)
        scores = check_relevancy(query, context, answer)

        if scores["groundedness"] < 0.5 or scores["relevance"] < 0.5:
            answer = (
                f"I couldn't find sufficient information in "
                f"{ref_type} {direct_result['ref_id']} to answer confidently.\n\n"
                f"Attempt:\n" + answer
            )

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": [{
                "type": ref_type,
                "paper": payload.get("paper_title"),
                "section": payload.get("section"),
                "page": payload.get("page"),
                "id": direct_result["ref_id"],
                "scope": "direct",
            }],
            "scores": scores,
            "history": history,
        }

    # ── Normal vector search ──────────────────────────────────────────
    scope = detect_scope(contextualized, paper_title)
    print(f"   Scope: {scope}")

    if scope == "local":
        nodes = retrieve(contextualized, top_k=top_k, paper_title=paper_title)
        if not nodes or nodes[0].score < 0.4:
            print("   Local weak — falling back to global")
            scope = "global"
            nodes = retrieve(contextualized, top_k=top_k)
    else:
        nodes = retrieve(contextualized, top_k=top_k, paper_titles=paper_titles)

    nodes = deduplicate_nodes(nodes)
    print(f"   Retrieved {len(nodes)} chunks ({scope})")

    enriched = enrich_with_references(nodes)
    context = build_context(enriched)
    answer = generate_answer(query, context, history)
    scores = check_relevancy(query, context, answer)

    if scores["groundedness"] < 0.5 or scores["relevance"] < 0.5:
        answer = (
            "I don't have sufficient information to answer this confidently.\n\n"
            "Original attempt:\n" + answer
        )

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    sources = [
        {
            "paper": n["metadata"].get("paper_title"),
            "section": n["metadata"].get("section"),
            "page": n["metadata"].get("page"),
            "type": n["metadata"].get("type"),
            "score": round(n["score"], 3),
            "scope": scope,
        }
        for n in enriched
    ]

    return {
        "answer": answer,
        "sources": sources,
        "scores": scores,
        "scope": scope,
        "history": history,
    }


def query_figure_with_question(pil_image, question, caption):
    """Re-query VLM on figure with user's specific question."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
    res = ollama_client.chat(model='qwen2.5vl:7b', messages=[{
        'role': 'user',
        'content': f"""Figure caption: {caption if caption else 'not provided'}

Specific question about this figure: {question}

Answer only based on what you can directly observe in the image.
Do not infer or add information not visible.""",
        'images': [img_base64]
    }])
    return res['message']['content']


if __name__ == "__main__":
    print("📚 Research RAG — ready")
    print("=" * 50)

    history = []

    # Simple test question
    query = "What method does EEGformer use for classification, and what method does BrainBERT use? Describe each separately."
    paper_titles = None
    paper_titles = ["EEGformer", "BrainBert"]

    result = chat(query, history=history, paper_titles=paper_titles)

    print(f"\n💬 Question: {query}")
    print(f"\n Paper filter: {paper_titles}")
    print(f"\n🤖 Answer:\n{result['answer']}")
    print(f"\n📎 Sources:")
    for s in result["sources"]:
        print(f"   - [{s['type']}] {s['paper']} | {s['section']} | page {s['page']} (score: {s['score']})")
    print(f"\n📊 Quality scores: {result['scores']}")
    a = 'animal'
    history = result['history'][:-1]