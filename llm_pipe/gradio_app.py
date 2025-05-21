import os
import gradio as gr
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import re
import requests
from gtts import gTTS
import tempfile
import json
import platform
from PIL import Image
from collections import defaultdict
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from llm_pipe.Ingestion_Retrieval.pinecone_retrieval import init_pinecone_and_embeddings, process_query
# import YOUTUBE QA AGENT FILE
from llm_pipe.Ingestion_Retrieval.youtube_qa_agent import handle_url_submit, answer_question, summarize_transcript


# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("api_key")
INDEX_NAME = "document-analysis"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Embedding model

# Configure Ollama API - use localhost as Ollama should be running locally
OLLAMA_API_BASE = "http://localhost:11434/api"
OLLAMA_MODEL = "phi3:mini"  # Ollama model name

# Detect OS for better error messages
is_windows = platform.system() == "Windows"
is_wsl = "microsoft" in platform.uname().release.lower() or "wsl" in platform.uname().release.lower()

# Company to PDF mapping - maps company names to their respective PDF files
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

def check_ollama_available():
    """Check if Ollama is available by sending a request to list models"""
    try:
        print(f"Trying to connect to Ollama at {OLLAMA_API_BASE}...")
        response = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            available_models = [model["name"] for model in models.get("models", [])]
            if OLLAMA_MODEL in available_models:
                print(f"Ollama is available with {OLLAMA_MODEL}")
                return True
            else:
                print(f"Warning: {OLLAMA_MODEL} not found in Ollama. Available models: {available_models}")
                return False
        else:
            print(f"Ollama API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Failed to connect to Ollama: {e}")
        return False

# Initialize Pinecone and embedding models using the imported function
pc, index, embeddings_model = init_pinecone_and_embeddings()
ollama_available = check_ollama_available()

def format_documents_with_page_info(results):
    """
    Format retrieved documents for inclusion in the context, 
    including clear page number formatting for the LLM to recognize.
    """
    if not results.get("results"):
        return "No relevant information found."
    
    formatted_text = ""
    
    # Extract the document-page mapping for reference
    doc_page_mapping = results.get("doc_page_mapping", {})
    
    # Process results by document and page
    for doc_id, pages in results.get("results", {}).items():
        formatted_text += f"\n=== Document: {doc_id} ===\n"
        
        for page_num, matches in pages.items():
            formatted_text += f"\n[Page {page_num}]\n"
            
            for match in matches:
                content_type = "Table" if match.get("table_info") else "Text"
                formatted_text += f"--- {content_type}{match.get('table_info', '')} (Relevance score: {match.get('score', 0):.2f}) ---\n"
                formatted_text += match.get("text", "").strip() + "\n\n"
    
    return formatted_text

def generate_answer_with_ollama(context: str, query: str, doc_page_mapping: Dict) -> str:
    """Generate an answer using Ollama's Phi-3 model based on context and query,
    with instructions to include page references."""
    
    # Format the doc-page mapping for inclusion in the prompt
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
    
    # Make API call to Ollama
    try:
        response = requests.post(
            f"{OLLAMA_API_BASE}/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 512
                }
            },
            timeout=600  # Increase timeout to 180 seconds
        )
        
        if response.status_code == 200:
            return response.json().get("response", "Error generating response.")
        else:
            return f"Error: Ollama returned status code {response.status_code}"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

def process_query_and_generate(company: str, query: str) -> str:
    """Process a query for a specific company and generate answer with page references."""
    if not query.strip():
        return "Please enter a query."
    
    # Use the imported process_query function to get search results
    results = process_query(index, embeddings_model, company, query)
    
    # Format retrieved documents with clear page information
    context = format_documents_with_page_info(results)
    
    # Get document-page mapping for reference in the prompt
    doc_page_mapping = results.get("doc_page_mapping", {})
    
    # Generate answer using Phi-3 via Ollama with instructions to include page references
    if "No relevant information found" in context:
        answer = "I couldn't find relevant information to answer your question in the selected document."
    else:
        try:
            answer = generate_answer_with_ollama(context, query, doc_page_mapping)
            
            # Post-process answer to highlight page citations
            answer = re.sub(r'\[Page (\d+)\]', r'**[Page \1]**', answer)
        except Exception as e:
            answer = f"Error: Unable to connect to Ollama. Please make sure Ollama is installed and running with the phi3:mini model."
    
    return answer, doc_page_mapping

def create_interface():
    """Create unified Gradio interface with two tabs: Document QA + YouTube QA."""
    # Create list of PDF options for dropdown
    pdf_options = []
    for company, pdfs in COMPANY_PDF_MAPPING.items():
        for pdf in pdfs:
            pdf_options.append(f"{company.capitalize()}: {pdf}")

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Tabs():

            # === Tab 1: Document QA System ===
            with gr.Tab("📄 Document QA System"):
                gr.Markdown("# 📊 Financial Reports QA\n**Extract valuable insights from SEC filings and annual reports**")

                if not ollama_available:
                    gr.Markdown("⚠️ **Ollama is not available.** Ensure it is running and the Phi-3 model is pulled.")

                with gr.Row():  # Full row split in two halves
                    with gr.Column(scale=1, min_width=600):  # Left side
                        with gr.Column():
                            gr.Markdown("### 📁 Select Report & Ask Your Question")

                            pdf_dropdown = gr.Dropdown(
                                choices=pdf_options,
                                label="Select Financial Report",
                                value=pdf_options[0] if pdf_options else None,
                                interactive=True
                            )

                            query_input = gr.Textbox(
                                lines=3,
                                label="Your Question",
                                placeholder="Example: What were the total R&D expenses for 2022?"
                            )

                            submit_btn = gr.Button("Get Answer", variant="primary")

                        with gr.Column():
                            gr.Markdown("### 🧠 Answer from the AI")

                            answer_output = gr.Textbox(
                                label="Answer",
                                lines=10,
                                show_copy_button=True
                            )

                            speak_btn = gr.Audio(label="Click Play to Hear", type="filepath")

                    with gr.Column(scale=1, min_width=600):  # Right side
                        with gr.Column():
                            gr.Markdown("### 📄 Pages Cited in Answer")

                            citation_gallery = gr.Gallery(
                                label="Source Pages",
                                show_label=True,
                                columns=1,
                                height="600px"
                            )

                # Handle submission
                def on_submit(pdf_selection, query):
                    if not pdf_selection:
                        return "Please select a financial report first.", None, []

                    company = pdf_selection.split(":")[0].lower()
                    answer, doc_page_mapping = process_query_and_generate(company, query)

                    # Load page images using the mapping
                    image_paths = []
                    for doc_id, pages in doc_page_mapping.items():
                        pdf_name = COMPANY_PDF_MAPPING[company][0]
                        image_map_file = f"{pdf_name}_images.json"
                        if os.path.exists(image_map_file):
                            with open(image_map_file, "r") as f:
                                image_map = json.load(f)
                                for page in pages:
                                    path = image_map.get(page)
                                    if path and os.path.exists(path):
                                        image_paths.append(Image.open(path))

                    # Convert answer to speech using gTTS
                    tts = gTTS(answer)
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tts.save(temp_file.name)

                    return answer, temp_file.name, image_paths

                submit_btn.click(
                    fn=on_submit,
                    inputs=[pdf_dropdown, query_input],
                    outputs=[answer_output, speak_btn, citation_gallery]
                )

            # === Tab 2: YouTube QA Agent ===
            with gr.Tab("🎥 YouTube QA Agent"):
                gr.Markdown("## 🎥 YouTube QA Agent (Powered by Ollama)")

                with gr.Row():
                    url_input = gr.Textbox(label="YouTube URL")
                    model_selector = gr.Dropdown(choices=["mistral", "llama3", "gemma"], label="Ollama Model", value="mistral")
                    fetch_button = gr.Button("Fetch Transcript")

                status_output = gr.Textbox(label="Status", interactive=False)
                fetch_button.click(fn=handle_url_submit, inputs=[url_input], outputs=status_output)

                gr.Markdown("### ❓ Ask Questions")
                with gr.Row():
                    question_input = gr.Textbox(label="Your Question")
                    ask_button = gr.Button("Ask")
                    answer_output = gr.Textbox(label="Answer", interactive=False)

                ask_button.click(fn=answer_question, inputs=[question_input, url_input, model_selector], outputs=answer_output)

                gr.Markdown("### 🧾 Summarize Video")
                with gr.Row():
                    summarize_button = gr.Button("Summarize")
                    summary_output = gr.Textbox(label="Summary", interactive=False)

                summarize_button.click(fn=summarize_transcript, inputs=[url_input, model_selector], outputs=summary_output)
                
    return demo


# Launch the Gradio app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)