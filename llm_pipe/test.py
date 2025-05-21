import os
import gradio as gr
from typing import Dict, List
from dotenv import load_dotenv
import requests
import platform
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from gradio import State
from llm_pipe.Ingestion_Retrieval.pinecone_retrieval import (
    init_pinecone_and_embeddings,
    semantic_search,
    format_documents,
    process_query,
    get_pdf_citation_info
)

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("api_key")
INDEX_NAME = "document-analysis"
EMBEDDING_MODEL = "all-mpnet-base-v2"

# Ollama API configuration
OLLAMA_API_BASE = "http://localhost:11434/api"
OLLAMA_MODEL = "phi3:mini"

# OS detection
is_windows = platform.system() == "Windows"
is_wsl = "microsoft" in platform.uname().release.lower() or "wsl" in platform.uname().release.lower()

# Company to PDF mapping
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
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            available_models = [model["name"] for model in models.get("models", [])]
            return OLLAMA_MODEL in available_models
        return False
    except:
        return False

def generate_answer_with_ollama(context: str, query: str) -> str:
    prompt = f"""You are a helpful assistant that answers questions about financial reports and SEC filings.
Answer the user's question based solely on the provided context.
If you don't know the answer based on the context, just say so.
Don't make up information and cite the relevant parts of the document when possible.

Context:
{context}

User Question:
{query}

Answer:"""

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
            timeout=600
        )
        if response.status_code == 200:
            return response.json().get("response", "Error generating response.")
        else:
            return f"Error: Ollama returned status code {response.status_code}"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

# Initialize Pinecone and embedding models
pc, index, embeddings_model = init_pinecone_and_embeddings()
ollama_available = check_ollama_available()


# def process_query_and_generate(company: str, query: str):
#     if not query.strip():
#         return "Please enter a query.", "", ""

#     results = process_query(index, embeddings_model, company, query)
#     context = format_documents(results)

#     if "No relevant information found" in context:
#         return "I couldn't find relevant information to answer your question in the selected document.", "", ""

#     answer = generate_answer_with_ollama(context, query)
#     citation = get_pdf_citation_info(results)

#     return answer, citation, answer
def process_query_and_generate(company: str, query: str):
    try:
        print(f"[INFO] Query received: {query}")
        results = process_query(index, embeddings_model, company, query)
        print(f"[INFO] Retrieval results: {results}")
        
        context = format_documents(results)
        print(f"[INFO] Context after formatting:\n{context}")

        if "No relevant information found" in context:
            return "I couldn't find relevant information to answer your question in the selected document.", "", ""

        answer = generate_answer_with_ollama(context, query)
        print(f"[INFO] Generated answer: {answer}")

        citation = get_pdf_citation_info(results)
        print(f"[INFO] Citation info: {citation}")

        return answer, citation, answer
    except Exception as e:
        print(f"[ERROR] Exception occurred: {str(e)}")
        return f"Error: {str(e)}", f"Error: {str(e)}", f"Error: {str(e)}"



def create_interface():
    pdf_options = [f"{company.capitalize()}: {pdf}" for company, pdfs in COMPANY_PDF_MAPPING.items() for pdf in pdfs]

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📊 Financial Reports QA\nExtract valuable insights from SEC filings and annual reports")

        if not ollama_available:
            gr.Markdown("⚠️ **Ollama is not available.** Ensure it is running and the Phi-3 model is pulled.")

        pdf_dropdown = gr.Dropdown(
            choices=pdf_options,
            label="Select Financial Report",
            value=pdf_options[0] if pdf_options else None
        )
        query_input = gr.Textbox(
            label="Your Question",
            placeholder="Example: What were the R&D expenses in 2022?",
            lines=3
        )
        submit_btn = gr.Button("Get Answer", variant="primary")

        answer_output = gr.Textbox(label="Answer", lines=10, show_copy_button=True)
        citation_output = gr.Textbox(label="Citation", visible=True)
        audio_output = gr.Audio(label="Spoken Answer", visible=True, type="filepath")
        
        citation_popup_visible = State(False)


        def toggle_citation_popup(citation, visible):
            return (
                gr.update(value=f"**Citation**\n\n{citation}", visible=not visible),
                not visible
            )


        def on_submit(pdf_selection, query):
            if not pdf_selection:
                return "Please select a financial report first.", "", ""
            company = pdf_selection.split(":")[0].lower()
            return process_query_and_generate(company, query)

        submit_btn.click(fn=on_submit, inputs=[pdf_dropdown, query_input], outputs=[answer_output, citation_output, audio_output])

        with gr.Row():
            speak_button = gr.Button("🔊 Speak")
            citation_button = gr.Button("📄 Show Citation")
            citation_markdown = gr.Markdown(visible=False)

        def speak(text):
            from gtts import gTTS
            import tempfile
            import os

            tts = gTTS(text)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_file.name)
            temp_file.close()
            return temp_file.name  # Gradio will play this file if it's a valid path


        speak_button.click(speak, inputs=[answer_output], outputs=audio_output)
        # citation_button.click(
        #     fn=lambda citation: gr.update(value=f"**Citation:**\n\n{citation}", visible=True),
        #     inputs=[citation_output],
        #     outputs=[citation_markdown],
        #     show_progress=False
        # )

        citation_button.click(
            fn=toggle_citation_popup,
            inputs=[citation_output, citation_popup_visible],
            outputs=[citation_markdown, citation_popup_visible],
            show_progress=False
        )

        gr.Markdown("<div style='text-align:center; font-size:0.9em; color:gray;'>Powered by Pinecone and Ollama (Phi-3) | Financial QA v1.0</div>")

    return demo
# def create_interface():
#     pdf_options = [f"{company.capitalize()}: {pdf}" for company, pdfs in COMPANY_PDF_MAPPING.items() for pdf in pdfs]

#     with gr.Blocks(theme=gr.themes.Soft()) as demo:
#         gr.Markdown("# 📊 Financial Reports QA\nExtract valuable insights from SEC filings and annual reports")

#         if not ollama_available:
#             gr.Markdown("⚠️ **Ollama is not available.** Ensure it is running and the Phi-3 model is pulled.")

#         pdf_dropdown = gr.Dropdown(
#             choices=pdf_options,
#             label="Select Financial Report",
#             value=pdf_options[0] if pdf_options else None
#         )
#         query_input = gr.Textbox(
#             label="Your Question",
#             placeholder="Example: What were the R&D expenses in 2022?",
#             lines=3
#         )
#         submit_btn = gr.Button("Get Answer", variant="primary")

#         answer_output = gr.Textbox(label="Answer", lines=10, show_copy_button=True)
#         citation_output = gr.Textbox(label="Citation", visible=True)
#         audio_output = gr.Audio(label="Spoken Answer", visible=True, type="filepath")
        
#         citation_popup_visible = State(False)

#         with gr.Row():
#             speak_button = gr.Button("🔊 Speak")
#             citation_button = gr.Button("📄 Show Citation")
#             citation_markdown = gr.Markdown(visible=False)

#         def on_submit(pdf_selection, query):
#             if not pdf_selection:
#                 return "Please select a financial report first.", "", ""
#             company = pdf_selection.split(":")[0].lower()
#             return process_query_and_generate(company, query)

#         submit_btn.click(
#             fn=on_submit,
#             inputs=[pdf_dropdown, query_input],
#             outputs=[answer_output, citation_output, audio_output]
#         )

#         def speak(text):
#             from gtts import gTTS
#             import tempfile
#             import os

#             tts = gTTS(text)
#             temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
#             tts.save(temp_file.name)
#             temp_file.close()
#             return temp_file.name

#         speak_button.click(
#             fn=speak,
#             inputs=[answer_output],
#             outputs=[audio_output]
#         )

#         citation_button.click(
#             fn=lambda citation: gr.update(value=f"**Citation:**\n\n{citation}", visible=True),
#             inputs=[citation_output],
#             outputs=[citation_markdown],
#             show_progress=False
#         )

#         gr.Markdown("<div style='text-align:center; font-size:0.9em; color:gray;'>Powered by Pinecone and Ollama (Phi-3) | Financial QA v1.0</div>")

#     return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)