import os
import gradio as gr
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import re
import requests
import json
import platform
from collections import defaultdict
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from llm_pipe.Ingestion_Retrieval.pinecone_retrieval import init_pinecone_and_embeddings, process_query

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("api_key")
INDEX_NAME = "document-analysis"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Embedding model

OLLAMA_API_BASE = "http://localhost:11434/api"
OLLAMA_MODEL = "phi3:mini"

is_windows = platform.system() == "Windows"
is_wsl = "microsoft" in platform.uname().release.lower() or "wsl" in platform.uname().release.lower()

COMPANY_PDF_MAPPING = {
    "enersys": ["EnerSys-2023-10K.pdf", "EnerSys-2017-10K.pdf"],
    "amazon": ["Amazon10k2022.pdf"],
    "apple": ["Apple_10-K-2021.pdf"],
    "nvidia": ["Nvidia.pdf"],
    "tesla": ["Tesla.pdf"],
    "lockheed": ["Lockheed_martin_10k.pdf"],
    "advent": ["Advent_Technologies_2022_10K.pdf"],
    "transdigm": ["TransDigm-2022-10K.pdf"]
}

pc, index, embeddings_model = init_pinecone_and_embeddings()

def check_ollama_available():
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            return OLLAMA_MODEL in [model["name"] for model in models.get("models", [])]
    except:
        pass
    return False

ollama_available = check_ollama_available()

def format_documents_with_page_info(results):
    formatted_text = ""
    for doc_id, pages in results.get("results", {}).items():
        formatted_text += f"\n=== Document: {doc_id} ===\n"
        for page_num, matches in pages.items():
            formatted_text += f"\n[Page {page_num}]\n"
            for match in matches:
                content_type = "Table" if match.get("table_info") else "Text"
                formatted_text += f"--- {content_type}{match.get('table_info', '')} (Relevance score: {match.get('score', 0):.2f}) ---\n"
                formatted_text += match.get("text", "").strip() + "\n\n"
    return formatted_text or "No relevant information found."

def generate_answer_with_ollama(context: str, query: str, doc_page_mapping: Dict) -> str:
    page_references = ""
    for doc_id, pages in doc_page_mapping.items():
        page_references += f"- Document '{doc_id}' has relevant information on pages: {', '.join(pages)}\n"

    prompt = f"""You are a helpful assistant that answers questions about financial reports and SEC filings.
Answer the user's question based solely on the provided context.

IMPORTANT INSTRUCTIONS:
1. Always cite the specific page numbers when referencing information in your answer.
2. Format page citations as [Page X] within your answer text.
3. If information comes from multiple pages, cite each relevant page.
4. If you don't know the answer based on the context, just say so.
5. Don't make up information.

The following pages contain relevant information for this query:
{page_references}

Context:
{context}

User Question:
{query}

Answer (remember to cite specific page numbers in your response):"""

    try:
        response = requests.post(
            f"{OLLAMA_API_BASE}/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=180
        )
        return response.json().get("response", "Error generating response.")
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

# def get_first_page_image(doc_page_mapping):
#     if not doc_page_mapping:
#         return None
#     doc_id = list(doc_page_mapping.keys())[0]
#     page_number = doc_page_mapping[doc_id][0]
#     json_path = f"{doc_id}.pdf_images.json"
#     if os.path.exists(json_path):
#         with open(json_path, "r") as f:
#             image_map = json.load(f)
#             return image_map.get(str(page_number))
#     return None

def get_all_cited_images(doc_page_mapping):
    images = []
    for doc_id, pages in doc_page_mapping.items():
        json_path = f"{doc_id}.pdf_images.json"
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                image_map = json.load(f)
                for page in pages:
                    img = image_map.get(str(page))
                    if img:
                        images.append(img)
    return images


def process_query_and_generate(company: str, query: str):
    if not query.strip():
        return "Please enter a query.", None
    results = process_query(index, embeddings_model, company, query)
    context = format_documents_with_page_info(results)
    doc_page_mapping = results.get("doc_page_mapping", {})
    if "No relevant information found" in context:
        return "I couldn't find relevant information.", None
    answer = generate_answer_with_ollama(context, query, doc_page_mapping)
    answer = re.sub(r'\[Page (\d+)\]', r'**[Page \1]**', answer)
    image_path = get_first_page_image(doc_page_mapping)
    return answer, image_path

def create_interface():
    pdf_options = [f"{c.capitalize()}: {f}" for c, fs in COMPANY_PDF_MAPPING.items() for f in fs]
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Column():
            gr.Markdown('# Financial Reports QA')
            gr.Markdown('Extract valuable information from financial reports with page references')

            if not ollama_available:
                gr.Markdown('⚠️ **Ollama not available. Make sure to run Ollama and pull phi3:mini model.**')

            with gr.Row():
                pdf_dropdown = gr.Dropdown(choices=pdf_options, label="Financial Report")
                query_input = gr.Textbox(lines=3, label="Your Question")
                submit_btn = gr.Button("Get Answer", variant="primary")

            answer_output = gr.Textbox(label="Answer", lines=8, show_copy_button=True)
            image_output = gr.Image(label="Cited Page Image")

            def on_submit(pdf_selection, query):
                if not pdf_selection:
                    return "Please select a financial report.", None
                company = pdf_selection.split(":")[0].lower()
                return process_query_and_generate(company, query)

            submit_btn.click(on_submit, inputs=[pdf_dropdown, query_input], outputs=[answer_output, image_output])

            gr.Markdown('Powered by Pinecone + Phi-3 | **v1.0**')
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
