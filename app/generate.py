import base64
import json
from locale import normalize
from pathlib import Path
import ollama
from collections import defaultdict
import re
import io
from retrieval import resolve_paper_title, handle_direct_query, hybrid_retrieve, deduplicate_nodes, enrich_with_references

# --- CONFIG ---
COLLECTION = "research_papers"
OLLAMA_BASE_URL = "http://host.docker.internal:11434"


SYSTEM_PROMPT = """You are a research assistant helping with academic literature review.
You are given retrieved excerpts from research papers with the user query.
You should use the retrieved excerpts to answer the user's question as best as possible.

Rules:
- If multiple papers say different things, highlight the contradiction
- If the retrieved context doesn't answer the question, say so explicitly
- Try to keep it concise and focused on the question asked (max 300 words)
- Never make up information not present in the retrieved context"""

DEFAULT_CONTEXT_PROMPT = """You are a search expert. Rewrite the user's question into a professional academic search query. 
If there is conversation history, resolve all pronouns (it, they, this). 
If there is no history, expand the query with relevant technical keywords to improve retrieval from scientific papers.
Output only the rewritten query, nothing else."""

COMPARISON_CONTEXT_PROMPT = """Rewrite the question into a search query designed to find contrasting data. 
Ensure the query includes words like 'comparison', 'benchmarks', 'metrics', or 'performance difference'. 
If history exists, name the specific models/papers being compared.
Output only the rewritten query, nothing else."""

GAP_CONTEXT_PROMPT = """Rewrite the question into a search query that targets limitations and unexplored areas. 
Include keywords like 'limitations', 'future work', 'open problems', 'assumptions', or 'conclusions'. 
Output only the rewritten query, nothing else."""

HYPOTHESIS_CONTEXT_PROMPT = """Rewrite the question into a search query that looks for scientific evidence or correlations. 
Include keywords like 'experimental results', 'observed patterns', 'causality', or 'statistical significance'.
Output only the rewritten query, nothing else."""

TREND_CONTEXT_PROMPT = """Rewrite the question into a search query focused on the evolution of a field. 
Include keywords like 'state of the art', 'historical progression', 'emerging trends', or 'chronological shifts'.
Output only the rewritten query, nothing else."""

MODE_TRIGGERS = {
    "comparison": ["compare", "contrast", "difference", "vs", "versus", "both"],
    "gap":        ["gap", "limitation", "missing", "future work", "unexplored", "weakness"],
    "hypothesis": ["hypothesis", "suggest", "propose", "could", "what if", "predict"],
    "trend":      ["trend", "pattern", "over time", "evolution", "progress", "state of the art"],
}

CONTEXT_PROMPTS = {
    "default":    DEFAULT_CONTEXT_PROMPT,
    "comparison": COMPARISON_CONTEXT_PROMPT,
    "gap":        GAP_CONTEXT_PROMPT,
    "hypothesis": HYPOTHESIS_CONTEXT_PROMPT,
    "trend":      TREND_CONTEXT_PROMPT,
}


GLOBAL_TRIGGERS = [
    "other papers", "any papers", "across papers", "different papers", "various papers",
]

LOCAL_TRIGGERS = [
    "this paper", "the paper", "this study", "the study",
    "in this section", "in this part", "in this figure", "in this table", "in this equation",
]

def detect_scope(query, paper_title=None, paper_titles=None):
    """Determine if query needs local or global search."""
    if paper_title is None and paper_titles is None:
        return "global"
    query_lower = query.lower()
    if any(trigger in query_lower for trigger in GLOBAL_TRIGGERS):
        return "global"
    if any(trigger in query_lower for trigger in LOCAL_TRIGGERS):
        return "local"
    # check if user named a specific paper
    focused = ([paper_title] if paper_title else []) + (paper_titles or [])
    if any(title.lower() in query_lower for title in focused):
        return "local"
    return "local" if (paper_title or paper_titles) else "global"


def detect_mode(query):
    query_lower = query.lower()
    for mode, triggers in MODE_TRIGGERS.items():
        if any(t in query_lower for t in triggers):
            return mode
    return "default"


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

        Respond in JSON format only, no explanation:
        {{"groundedness": 0.5, "relevance": 0.5}}"""
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
    # 1. Detect the mode based on the user's raw query
    mode = detect_mode(query) 
    system_instruction = CONTEXT_PROMPTS[mode]
    print(f"   Contextualizing for mode: {mode}")
    history_context = json.dumps(history[-4:], indent=2) if history else "No previous conversation history."
    ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
    res = ollama_client.chat(model='qwen2.5:7b', messages=[
        {
            'role': 'system', 
            'content': system_instruction
        },
        {
            'role': 'user',
            'content': f"CONVERSATION HISTORY:\n{history_context}\n\nUSER QUESTION: {query}"
        }
    ])
    rewritten_query = res['message']['content'].strip()
    print(f"🔍 Original: {query}")
    print(f"🚀 Optimized Query: {rewritten_query}")
    return rewritten_query


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

def diversify_nodes(nodes, all_candidates, keyword_nodes, score_threshold=0.4, paper_title=None, paper_titles=None, scope="local"):
    if paper_title:
        return nodes

    covered = {n.metadata.get("paper_title") for n in nodes}
    
    if scope == "global":
        target_papers = {
            n.metadata.get("paper_title")
            for n in all_candidates
        }
    elif paper_titles:
        target_papers = set(paper_titles)

    missing = target_papers - covered

    if not missing:
        return nodes

    result = list(nodes)
    seen_ids = {n.node_id for n in nodes}
    keyword_ids = {
        n.get("id")
        for n in keyword_nodes
    }

    for paper in missing:
        best = next((n for n in sorted(all_candidates, key=lambda n: n.score, reverse=True)
                     if n.metadata.get("paper_title") == paper), None)
        if best and best.score >= score_threshold and best.node_id not in seen_ids and best.node_id not in keyword_ids:
            result.append(best)
            seen_ids.add(best.node_id)

    result.sort(key=lambda n: n.score, reverse=True)
    return result


def chat_retrieve(query, history=[], paper_title=None, paper_titles=None, top_k=5):
    contextualized = contextualize_query(query, history)
    print(f"\n🔍 Searching for: {contextualized}")

    if paper_title:
        paper_title = resolve_paper_title(paper_title)
        print(f"   Resolved paper title: {paper_title}")

    if paper_titles:
        paper_titles = [resolve_paper_title(t) for t in paper_titles]
        print(f"   Resolved paper titles: {paper_titles}")
    
    # ── Direct lookup for figure/table/equation queries ───────────────
    direct_result, ref_type = handle_direct_query(query, paper_title)
    if direct_result:
        payload = direct_result["payload"]
        referring_texts = direct_result["referring_texts"]
        pil_image = direct_result.get("pil_image")

        print(f"   Direct {ref_type} lookup — bypassing vector search")
        print(f"   Found in: {payload.get('paper_title')}")
        print(f"   {len(referring_texts)} referring text chunks found")

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

        if ref_type == "figure" and pil_image:
            print("   Re-querying VLM with specific question...")
            vlm_answer = query_figure_with_question(pil_image, query, payload.get("caption", ""))
            context += f"\n\n---\n\n[VLM ANALYSIS FOR QUERY]\n{vlm_answer}"

        answer = generate_answer(query, context, history)
        scores = check_relevancy(query, context, answer)

        return {
            "answer": answer,
            "sources": [{
                "type": ref_type,
                "paper": payload.get("paper_title"),
                "section": payload.get("section"),
                "page": payload.get("page"),
                "id": direct_result["ref_id"],
                "scope": "direct",
                "node_text": json.loads(payload.get("_node_content", "{}")).get("text", ""),
                "filename": Path(payload.get("source_pdf", "")).name,
                "source_pdf": payload.get("source_pdf"),
                "bbox_l": payload.get("bbox_l"),
                "bbox_t": payload.get("bbox_t"),
                "bbox_r": payload.get("bbox_r"),
                "bbox_b": payload.get("bbox_b"),
            }],
            "scores": scores,
            "history": history,
        }

    # ── Normal vector search ──────────────────────────────────────────
    scope = detect_scope(query, paper_title, paper_titles)
    print(f"   Scope: {scope}")

    if scope == "local":
        semantic_nodes, keyword_nodes, nodes = hybrid_retrieve(
            contextualized, top_k=top_k, paper_title=paper_title, paper_titles=paper_titles
        )
    else:
        semantic_nodes, keyword_nodes, nodes = hybrid_retrieve(
            contextualized, top_k=top_k
        )
    semantic_nodes = deduplicate_nodes(semantic_nodes)
    semantic_nodes = diversify_nodes(semantic_nodes, nodes, keyword_nodes=keyword_nodes, paper_title=paper_title, paper_titles=paper_titles, scope=scope)
    print(f"   Retrieved {len(semantic_nodes)} semantic + {len(keyword_nodes)} keyword chunks ({scope})")

    enriched = enrich_with_references(semantic_nodes)

    if keyword_nodes:
        keyword_context = "\n\n---\n\n".join([
            f"[KEYWORD MATCH | {n['metadata'].get('paper_title')} | "
            f"Section: {n['metadata'].get('section')} | Page: {n['metadata'].get('page')}]\n"
            f"{n['text']}"
            for n in keyword_nodes
        ])
        context = build_context(enriched) + "\n\n---\n\n" + keyword_context
    else:
        context = build_context(enriched)

    answer = generate_answer(query, context, history)
    scores = check_relevancy(query, context, answer)

    sources = [
        {
            "paper": n["metadata"].get("paper_title"),
            "section": n["metadata"].get("section"),
            "page": n["metadata"].get("page"),
            "type": n["metadata"].get("type"),
            "node_text": n["text"],
            "filename": Path(n["metadata"].get("source_pdf", "")).name,
            "source_pdf": n["metadata"].get("source_pdf"),
            "bbox_l": n["metadata"].get("bbox_l"),
            "bbox_t": n["metadata"].get("bbox_t"),
            "bbox_r": n["metadata"].get("bbox_r"),
            "bbox_b": n["metadata"].get("bbox_b"),
            "score": round(n["score"], 3),
            "scope": scope,
        }
        for n in enriched
    ]
    sources.extend([
        {
        "paper": n["metadata"].get("paper_title"),
        "section": n["metadata"].get("section"),
        "page": n["metadata"].get("page"),
        "type": n["metadata"].get("type"),
        "node_text": n["text"],
        "filename": Path(n["metadata"].get("source_pdf", "")).name,
        "source_pdf": n["metadata"].get("source_pdf"),
        "bbox_l": n["metadata"].get("bbox_l"),
        "bbox_t": n["metadata"].get("bbox_t"),
        "bbox_r": n["metadata"].get("bbox_r"),
        "bbox_b": n["metadata"].get("bbox_b"),
        "score": None,  # keyword retrieval doesn't use embedding score
        "scope": scope,
        "retrieval_type": "keyword"
        } 
    for n in keyword_nodes
    ])

    return {
        "answer": answer,
        "sources": sources,
        "scores": scores,
        "scope": scope,
        "history": history,
    }

    

def chat(query, history=[], paper_title=None, paper_titles=None, top_k=5):
    output = chat_retrieve(query, history=history, paper_title=paper_title, paper_titles=paper_titles, top_k=top_k)
    scores = output["scores"]
    if scores["groundedness"] < 0.5 or scores["relevance"] < 0.5:
        expanded_queries = expand_query(query)
        for eq in expanded_queries:
            eq_output = chat_retrieve(eq, history=history, paper_title=paper_title, paper_titles=None, top_k=top_k)
            if eq_output["scores"]["groundedness"] >= 0.5 and eq_output["scores"]["relevance"] >= 0.5:
                output = eq_output
                break
        else:
            output["answer"] = (
                    f"I couldn't find sufficient information to answer confidently, try rephrasing your question.\n\n"
                )

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": output["answer"]})
    output["history"] = history

    return output


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
    print("Commands:")
    print("  /paper <name>     — filter to a specific paper")
    print("  /papers <n1>, <n2> — filter to multiple papers")
    print("  /clear            — clear paper filter")
    print("  /history          — show conversation history")
    print("  /reset            — reset conversation")
    print("  /quit             — exit")
    print("=" * 50)

    history = []
    paper_title = None
    paper_titles = {"EEGFormer", "BrainBert"}

    while True:
        try:
            query = input("\n💬 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not query:
            continue

        # ── Commands ──────────────────────────────────────────────────
        if query.startswith("/quit"):
            print("Goodbye!")
            break

        elif query.startswith("/reset"):
            history = []
            paper_title = None
            paper_titles = None
            print("✓ Conversation reset")
            continue

        elif query.startswith("/clear"):
            paper_title = None
            paper_titles = None
            print("✓ Paper filter cleared — searching all papers")
            continue

        elif query.startswith("/history"):
            if not history:
                print("No history yet")
            else:
                for turn in history:
                    role = "You" if turn["role"] == "user" else "RAG"
                    print(f"\n{role}: {turn['content'][:200]}...")
            continue

        elif query.startswith("/paper "):
            paper_title = query[7:].strip()
            paper_titles = None
            print(f"✓ Filtering to: {paper_title}")
            continue

        elif query.startswith("/papers "):
            names = query[8:].strip()
            paper_titles = [p.strip() for p in names.split(",")]
            paper_title = None
            print(f"✓ Filtering to: {paper_titles}")
            continue

        # ── Chat ──────────────────────────────────────────────────────
        result = chat(
            query,
            history=history,
            paper_title=paper_title,
            paper_titles=paper_titles,
        )

        print(f"\n🤖 Answer:\n{result['answer']}")
        print(f"\n📎 Sources:")
        for s in result["sources"]:
            score = s.get('score', 'direct')
            print(f"   - [{s['type']}] {s['paper']} | {s['section']} | page {s['page']} (score: {score})")
        print(f"\n📊 Quality: groundedness={result['scores']['groundedness']} relevance={result['scores']['relevance']}")

        history = result["history"]