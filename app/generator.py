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
