import io
import re
import base64
from pathlib import Path
from sys import prefix
import ollama
from pix2tex.cli import LatexOCR
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from llama_index.core import Document as LlamaDocument, StorageContext, \
    VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from docling_core.types.doc.labels import DocItemLabel
from tqdm import tqdm

# --- CONFIG ---
# This path is where the PDFs will be INSIDE the Docker container
PDF_DIR = Path("/app/documents") 
# QDRANT_PATH = "/app/qdrant_db"
COLLECTION = "research_papers"
OLLAMA_BASE_URL = "http://host.docker.internal:11434"

# Initialize Nomic Embeddings via Ollama
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text-v2-moe",
    base_url=OLLAMA_BASE_URL
)

# # Initialize Qdrant Local
# client = QdrantClient(path=QDRANT_PATH)
# vector_store = QdrantVectorStore(collection_name=COLLECTION, client=client)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)

latex_model = LatexOCR()

TEXT_LABELS = (DocItemLabel.PARAGRAPH, DocItemLabel.TEXT)

# Initialize Docling
# Disable the heavy AI layout model
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True         
pipeline_options.do_table_structure = True  
pipeline_options.generate_page_images = True  # This fixes the NoneType image issue
pipeline_options.images_scale = 2.0           # Makes images clear for Qwen2.5-VL
converter = DocumentConverter(
    format_options={
        "pdf": PdfFormatOption(pipeline_options=pipeline_options)
    }
)

def get_provenance(item, pdf_path):
    if item.prov:
        return {
            "page": item.prov[0].page_no,
            "bbox_l": item.prov[0].bbox.l,
            "bbox_t": item.prov[0].bbox.t,
            "bbox_r": item.prov[0].bbox.r,
            "bbox_b": item.prov[0].bbox.b,
            "source_pdf": str(pdf_path),
        }
    return {"page": None, "bbox_l": None, "bbox_t": None, "bbox_r": None, "bbox_b": None, "source_pdf": str(pdf_path)}


def describe_figure(image_pil, caption):
    """Ollama Vision Chat for research figures."""
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
    
    try:
        res = ollama_client.chat(model='qwen2.5vl:7b', messages=[{
            'role': 'user',
            'content': f"""You are analyzing a figure from a scientific paper.
        Caption: {caption if caption else 'not provided — extract from image if visible.'}

        Describe only what you can directly observe:
        - Figure number and title if visible
        - What the figure shows (architecture, chart, diagram, results, etc.)
        - Key components, labels, or values visible in the image
        - Any trends or relationships shown

        Do not infer or add information not visible in the image.""",
                    'images': [img_base64]
                }])
        return res['message']['content']
    except Exception as e:
        return f"Vision analysis failed: {str(e)}"

def get_academic_title(doc):
    """
    Finds the 'heaviest' text on the first page.
    Corrected to use valid DocItemLabels.
    """
    candidates = []

    # These are the valid labels in the current docling-core
    # We look for TITLE, SECTION_HEADER, and PARAGRAPH (as fallbacks)
    target_labels = [DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER, DocItemLabel.PARAGRAPH]

    for item, level in doc.iterate_items():
        # 1. Only look at the first page
        if not item.prov or item.prov[0].page_no > 1:
            continue

        # 2. Check if the label is one we care about
        if item.label in target_labels:
            # Calculate height of the bounding box
            # docling uses a coordinate system where 'b' is bottom and 't' is top
            bbox = item.prov[0].bbox
            height = bbox.t - bbox.b
            
            # Filter out very small text (headers/DOIs/metadata)
            if height > 5: 
                candidates.append({
                    "text": item.text.strip(),
                    "height": height,
                    "top": bbox.t,
                    "label": item.label
                })

    if not candidates:
        return "Unknown_Title"

    # 3. Prioritize DocItemLabel.TITLE first
    # If no TITLE label exists, prioritize by font Height, then vertical Position
    candidates.sort(key=lambda x: (x["label"] == DocItemLabel.TITLE, x["height"], x["top"]), reverse=True)

    best_candidate = candidates[0]["text"]
    
    # Clean up: remove newlines often found in titles
    clean_title = " ".join(best_candidate.split())
    
    # Validation: Titles are usually between 3 and 50 words
    word_count = len(clean_title.split())
    if 2 < word_count < 50:
        return clean_title
    
    return "Unknown_Title"


def match_by_proximity(items, captions, max_distance=150):
    """Match items to captions by closest vertical midpoint proximity."""
    caption_map = {}
    remaining = list(captions)
    
    for item, _ in items:
        item_mid = (item.prov[0].bbox.t + item.prov[0].bbox.b) / 2
        
        best_caption = None
        best_distance = max_distance
        
        for caption in remaining:
            cap_mid = (caption.prov[0].bbox.t + caption.prov[0].bbox.b) / 2
            distance = abs(item_mid - cap_mid)
            if distance < best_distance:
                best_distance = distance
                best_caption = caption
        
        if best_caption:
            caption_map[id(item)] = best_caption.text.strip()
            remaining.remove(best_caption)
    
    return caption_map


def build_caption_map(items_list):
    caption_map = {}
    
    for page_no in set(item.prov[0].page_no for item, _ in items_list if item.prov):
        
        page_tables = sorted(
            [(item, idx) for idx, (item, _) in enumerate(items_list)
             if item.prov and item.prov[0].page_no == page_no 
             and item.label == DocItemLabel.TABLE],
            key=lambda x: x[0].prov[0].bbox.t,
            reverse=True
        )
        page_figures = sorted(
            [(item, idx) for idx, (item, _) in enumerate(items_list)
             if item.prov and item.prov[0].page_no == page_no 
             and item.label == DocItemLabel.PICTURE],
            key=lambda x: x[0].prov[0].bbox.t,
            reverse=True
        )
        page_captions = sorted(
            [item for item, _ in items_list
             if item.prov and item.prov[0].page_no == page_no
             and item.label == DocItemLabel.CAPTION],
            key=lambda x: x.prov[0].bbox.t,
            reverse=True
        )

        table_captions = [
            c for c in page_captions 
            if re.search(r'Table\s+\d+', c.text, re.IGNORECASE)
        ]
        figure_captions = [
            c for c in page_captions 
            if re.search(r'Fig(?:ure)?\.?\s*\d+', c.text, re.IGNORECASE)
        ]

        # Match main tables and figures
        caption_map.update(match_by_proximity(page_tables, table_captions))
        caption_map.update(match_by_proximity(page_figures, figure_captions))

    return caption_map


def process_research_paper(pdf_path):
    result = converter.convert(pdf_path)
    doc = result.document
    final_nodes = []

    academic_title = get_academic_title(doc)

    # ── 1. Pre-collect captions ───────────────────────────────────────────
    items_list = list(doc.iterate_items())
    caption_map = build_caption_map(items_list)

    # Build flat index of all paragraphs for formula context lookup
    all_para_texts = []
    for idx, (item, level) in enumerate(items_list):
        if item.label in TEXT_LABELS and item.text.strip():
            all_para_texts.append((idx, item.text))

    # ── 2. Group items into subsection buckets ────────────────────────────
    sections = []
    current_section = {"title": "Introduction", "items": []}

    for item, level in items_list:
        if item.label == DocItemLabel.SECTION_HEADER:
            sections.append(current_section)
            current_section = {"title": item.text.strip(), "items": []}
        else:
            current_section["items"].append(item)
    sections.append(current_section)

    # ── 3. Process each subsection ────────────────────────────────────────
    print(f"  Processing {len(sections)} sections...")
    for i, section in enumerate(sections):
        print(f"  Section {i+1}/{len(sections)}: {section['title'][:50]}")
        section_name = section["title"]
        section_text_pool = []
        section_equation_ids = []

        # Second pass: process all items
        for item in section["items"]:

            # ── TABLES ──────────────────────────────────────────────────
            if item.label == DocItemLabel.TABLE:
                caption = caption_map.get(id(item), "")
                label_match = re.search(r"(Table\s+\d+)", caption, re.IGNORECASE)
                table_label = label_match.group(1) if label_match else "Table"
                num = re.search(r'\d+', table_label)
                table_id = f"table_{num.group()}" if num else "table_unknown"

                try:
                    md_table = item.export_to_markdown(doc)
                except Exception:
                    md_table = "[Table export failed]"

                final_nodes.append(TextNode(
                    text=f"{table_label}\nCAPTION: {caption}\n{md_table}",
                    metadata={
                        "type": "table",
                        "paper_title": academic_title,
                        "section": section_name,
                        "table_id": table_id,
                        "caption": caption,
                        **get_provenance(item, pdf_path),
                    }
                ))

            # ── FIGURES ─────────────────────────────────────────────────
            elif item.label == DocItemLabel.PICTURE:
                caption = caption_map.get(id(item), "")
                label_match = re.search(r"(Fig(?:ure)?\.?\s*\d+)", caption, re.IGNORECASE) if caption else None

                pil_image = None
                try:
                    pil_image = item.get_image(doc)
                except Exception as e:
                    print(f"Could not get image: {e}")

                if pil_image is None:
                    print("Skipping figure — image is missing")
                    continue

                summary = describe_figure(pil_image, caption)

                if not label_match:
                    label_match = re.search(r"Fig(?:ure)?(?:\.?\s*|\s+number:?\s*)(\d+)", summary, re.IGNORECASE)

                if not label_match:
                    print("Skipping figure — could not determine figure number")
                    continue

                num_match = re.search(r'\d+', label_match.group(1))
                figure_label = f"Figure {num_match.group()}"
                figure_id = f"figure_{num_match.group()}"

                final_nodes.append(TextNode(
                    text=f"{figure_label}\nFIGURE ANALYSIS: {summary}\nCAPTION: {caption}",
                    metadata={
                        "type": "figure",
                        "paper_title": academic_title,
                        "section": section_name,
                        "figure_id": figure_id,
                        "caption": caption,
                        **get_provenance(item, pdf_path),
                    }
                ))

            # ── FORMULAS ────────────────────────────────────────────────
            elif item.label == DocItemLabel.FORMULA:
                text = item.text.strip()

                if not text:
                    try:
                        text = describe_formula(item.get_image(doc))
                    except Exception as e:
                        print(f"pix2tex failed: {e}")

                if not text:
                    print("Skipping formula — could not extract text or image")
                    continue

                eq_num = re.search(r'\((\d+)\)\s*[,.]?\s*$', text)
                if eq_num:
                    section_equation_ids.append(f"equation_{eq_num.group(1)}")
                eq_label = f"[Equation {eq_num.group(1)}]" if eq_num else "[Equation]"

                # surrounding has pre AND post context    
                formula_idx = next(i for i, (it, _) in enumerate(items_list) if it is item)
                surrounding = get_surrounding_paragraphs(formula_idx, all_para_texts, n=1)
                summary = summarize_formula(text, surrounding_context=surrounding)

                final_nodes.append(TextNode(
                    text=f"{eq_label}: {summary}",
                    metadata={
                        "type": "formula",
                        "latex": text,
                        "equation_id": f"equation_{eq_num.group(1)}" if eq_num else None,
                        "paper_title": academic_title,
                        "section": section_name,
                        **get_provenance(item, pdf_path),
                    }
                ))

            # ── PARAGRAPHS ───────────────────────────────────────────────
            elif item.label in TEXT_LABELS:
                if item.text.strip():
                    section_text_pool.append(item.text)

        # ── 4. Chunk section text ─────────────────────────────────────────
        if section_text_pool:
            
            full_section_text = "\n".join(section_text_pool)
            chunks = chunk_text_by_chars(full_section_text, limit=1200, overlap=200)

            has_formulas = any(item.label == DocItemLabel.FORMULA for item in section["items"])

            for chunk in chunks:
                table_refs = re.findall(r"(?:Supplementary\s+|Suppl\.\s+)?Table\s+\d+", chunk, re.IGNORECASE)
                fig_refs   = re.findall(r"(?:Supplementary\s+|Suppl\.\s+)?(?:Figure|Fig\.)\s*\d+", chunk, re.IGNORECASE)
                eq_refs    = re.findall(r"Equation\s+\d+", chunk, re.IGNORECASE)

                final_nodes.append(TextNode(
                    text=chunk,
                    metadata={
                        "type": "text",
                        "paper_title": academic_title,
                        "section": section_name,
                        "referenced_tables": normalize(table_refs),
                        "referenced_figures": normalize(fig_refs),
                        "referenced_equations": normalize(eq_refs) or section_equation_ids,  # still not perfect, coz sometimes the equations are a section of their own, so it might not refer
                        "has_formulas": has_formulas,
                        **(get_provenance(section["items"][0], pdf_path) if section["items"] else {}), # Assuming the section has at least one item
                    }
                ))

    return final_nodes


def normalize(refs):
    result = []
    for r in refs:
        num_match = re.search(r'\d+', r)
        if not num_match:
            continue
        num = num_match.group()
        is_supp = bool(re.search(r'Supplementary|Suppl\.', r, re.IGNORECASE))
        type_match = re.search(r'(?:Figure|Fig|Table|Equation|Eq)', r, re.IGNORECASE)
        if not type_match:
            continue
        prefix = type_match.group().lower()
        if 'fig' in prefix:
            prefix = 'figure'
        elif 'eq' in prefix:
            prefix = 'equation'
        else:
            prefix = 'table'
        if is_supp:
            result.append(f"supp_{prefix}_{num}")
        else:
            result.append(f"{prefix}_{num}")
    return list(set(result))


def describe_formula(image_pil):
    try:
        return latex_model(image_pil)
    except Exception as e:
        return ""


def get_surrounding_paragraphs(formula_idx, all_para_texts, n=2):
    """Get n paragraphs before and after the formula by doc position."""
    before = [text for idx, text in all_para_texts if idx < formula_idx][-n:]
    after  = [text for idx, text in all_para_texts if idx > formula_idx][:n]
    context = ""
    if before:
        context += "Text before formula:\n" + "\n".join(before)
    if after:
        context += "\n\nText after formula:\n" + "\n".join(after)
    return context

def summarize_formula(latex, surrounding_context=""):
    ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
    res = ollama_client.chat(model='qwen2.5:7b', messages=[{
        'role': 'user',
        'content': f"""Given this formula from a research paper:
{latex}

Context from surrounding text:
{surrounding_context}

In 2-3 sentences, explain what this formula computes and what each component represents.
Be specific, do not add anything not supported by the context."""
    }])
    return res['message']['content']


def chunk_text_by_chars(text, limit=1200, overlap=200):
    """Chunks text into character-limited segments while attempting to keep sentences whole."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + limit
        # If we aren't at the end, try to find a period to end on a sentence
        if end < text_len:
            last_period = text.rfind('.', start, end)
            if last_period != -1 and last_period > start + (limit // 2):
                end = last_period + 1
        
        chunks.append(text[start:end].strip())
        start = end - overlap # Overlap for context preservation
        
    return [c for c in chunks if c]