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
import pandas as pd
import sqlite3
from pathlib import Path
import time
from difflib import get_close_matches


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
    
    return formatted_text or "No relevant information found."

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
            timeout=180  # Increase timeout to 180 seconds
        )
        
        if response.status_code == 200:
            return response.json().get("response", "Error generating response.")
        else:
            return f"Error: Ollama returned status code {response.status_code}"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

#New
JSON_DIR = os.path.join(os.path.dirname(__file__), "..", "jsons")
def get_all_cited_images(doc_page_mapping):

    images = []
    for doc_id, pages in doc_page_mapping.items():
        json_filename = f"{doc_id}_images.json"
        json_path = os.path.join(JSON_DIR, json_filename)
        if os.path.exists(json_path):
            print(f"Loading JSON: {json_path}")
            with open(json_path, "r") as f:
                image_map = json.load(f)
                for page in pages:
                    img = image_map.get(str(page))
                    if img:
                        print(f"Found image path: {img}")    
                        images.append(img)
    return images

# def process_query_and_generate(company: str, query: str) -> str:
def process_query_and_generate(company: str, query: str) -> Tuple[str, Dict[str, List[str]], List[str]]:
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
            answer = re.sub(r'\[Page \d+\]', '', answer).strip()
        except Exception as e:
            answer = f"Error: Unable to connect to Ollama. Please make sure Ollama is installed and running with the phi3:mini model."
    
    image_paths = []
    for path in get_all_cited_images(doc_page_mapping):
        if os.path.exists(path):
            image_paths.append(path)
    return answer, doc_page_mapping, image_paths

# NEW SQL CODE START FROM HERE
# Function to get available models from Ollama
def get_available_models():
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            return [model['name'] for model in models.get('models', [])]
        return []
    except:
        return []

# Function to find closest column match
def find_closest_column(target_word, columns):
    """Find the closest matching column name using fuzzy matching"""
    # Direct match first
    if target_word in columns:
        return target_word
    
    # Try partial matches
    for col in columns:
        if target_word in col.lower() or col.lower() in target_word:
            return col
    
    # Use difflib for fuzzy matching
    matches = get_close_matches(target_word, columns, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    
    return None

# Enhanced SQL generation with better column matching
def generate_sql_with_ollama(query, table_name, columns, model=OLLAMA_MODEL):
    # Create a more detailed prompt with column information
    column_info = "\n".join([f"- {col}" for col in columns])
    
    prompt = f"""
You are an expert SQL query generator. Given a natural language question and database schema, generate a precise SQL query.

Database Information:
- Table name: {table_name}
- Available columns:
{column_info}

IMPORTANT RULES:
1. Generate ONLY the SQL query, no explanations or markdown
2. Use proper SQL syntax for SQLite
3. Column names must EXACTLY match the available columns listed above
4. Always use LIMIT clause for SELECT queries (default LIMIT 10)
5. For aggregation queries (COUNT, SUM, AVG, etc.), don't use LIMIT unless grouping
6. Use LIKE operator with % wildcards for text searches
7. If a column name in the question doesn't exist, find the closest matching column from the list above

Natural language question: {query}

SQL Query:"""

    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 150
            }
        }
        
        response = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json=payload,
            timeout=60  # Increased timeout to 60 seconds
        )
        
        if response.status_code == 200:
            result = response.json()
            sql_query = result['response'].strip()
            
            # Clean up the response
            sql_query = re.sub(r'```sql\n?', '', sql_query)
            sql_query = re.sub(r'```\n?', '', sql_query)
            sql_query = re.sub(r'^SQL Query:\s*', '', sql_query, flags=re.IGNORECASE)
            
            # Extract the first line that looks like a SQL query
            lines = sql_query.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.upper().startswith(('SELECT', 'PRAGMA', 'SHOW', 'DESC', 'WITH'))):
                    return line
            
            return sql_query.split('\n')[0].strip() if sql_query else None
            
        else:
            return None
            
    except requests.exceptions.Timeout:
        print("Ollama request timed out")
        return None
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None

# Enhanced fallback function with better column matching
def generate_sql_fallback(query, table_name, columns):
    query_lower = query.lower()
    
    # Extract potential column names from the query
    query_words = re.findall(r'\b\w+\b', query_lower)
    
    # For displaying all columns
    if any(phrase in query_lower for phrase in ["show all columns", "list all columns", "display all columns", "what columns", "which columns"]):
        return f"PRAGMA table_info({table_name})"
    
    # Find relevant columns mentioned in the query
    relevant_columns = []
    for word in query_words:
        closest_col = find_closest_column(word, columns)
        if closest_col and closest_col not in relevant_columns:
            relevant_columns.append(closest_col)
    
    # Detect query type and generate appropriate SQL
    if any(word in query_lower for word in ["average", "avg", "mean"]):
        if relevant_columns:
            # Try to find numeric columns
            numeric_cols = [col for col in relevant_columns if any(num_word in col.lower() for num_word in ['price', 'cost', 'amount', 'value', 'salary', 'age', 'count', 'number', 'score', 'rating'])]
            if numeric_cols:
                return f"SELECT AVG({numeric_cols[0]}) as average_{numeric_cols[0]} FROM {table_name}"
            else:
                return f"SELECT AVG({relevant_columns[0]}) as average_{relevant_columns[0]} FROM {table_name}"
        return f"SELECT * FROM {table_name} LIMIT 5"
    
    elif any(word in query_lower for word in ["sum", "total"]):
        if relevant_columns:
            numeric_cols = [col for col in relevant_columns if any(num_word in col.lower() for num_word in ['price', 'cost', 'amount', 'value', 'salary', 'count', 'number', 'score'])]
            if numeric_cols:
                return f"SELECT SUM({numeric_cols[0]}) as total_{numeric_cols[0]} FROM {table_name}"
            else:
                return f"SELECT SUM({relevant_columns[0]}) as total_{relevant_columns[0]} FROM {table_name}"
        return f"SELECT * FROM {table_name} LIMIT 5"
    
    elif any(word in query_lower for word in ["maximum", "max", "highest", "top"]):
        if relevant_columns:
            return f"SELECT MAX({relevant_columns[0]}) as max_{relevant_columns[0]} FROM {table_name}"
        return f"SELECT * FROM {table_name} ORDER BY {columns[0]} DESC LIMIT 5"
    
    elif any(word in query_lower for word in ["minimum", "min", "lowest", "bottom"]):
        if relevant_columns:
            return f"SELECT MIN({relevant_columns[0]}) as min_{relevant_columns[0]} FROM {table_name}"
        return f"SELECT * FROM {table_name} ORDER BY {columns[0]} ASC LIMIT 5"
    
    elif "count" in query_lower:
        if relevant_columns:
            return f"SELECT COUNT({relevant_columns[0]}) as count_{relevant_columns[0]} FROM {table_name}"
        return f"SELECT COUNT(*) as total_count FROM {table_name}"
    
    elif any(word in query_lower for word in ["show", "display", "list", "all", "get", "find"]):
        limit = 10
        limit_match = re.search(r"(\d+)", query)
        if limit_match:
            limit = int(limit_match.group(1))
        
        if relevant_columns:
            return f"SELECT {', '.join(relevant_columns)} FROM {table_name} LIMIT {limit}"
        else:
            return f"SELECT * FROM {table_name} LIMIT {limit}"
    
    else:
        # Default: show sample data
        return f"SELECT * FROM {table_name} LIMIT 5"

# Function to create an SQLite database from a CSV file
def create_database_from_csv(csv_file_path):
    db_name = f"rag_database_{int(time.time())}.db"
    
    df = pd.read_csv(csv_file_path)
    # Clean column names: remove special characters and convert to lowercase
    original_columns = df.columns.tolist()
    df.columns = [re.sub(r'[^\w]', '_', col).lower().strip('_') for col in df.columns]
    
    conn = sqlite3.connect(db_name)
    table_name = "data"
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    
    return db_name, table_name, list(df.columns), original_columns

# Enhanced SQL execution with better error handling
def execute_sql_query(db_name, sql_query):
    try:
        conn = sqlite3.connect(db_name)
        
        if sql_query.strip().upper().startswith("PRAGMA"):
            cursor = conn.cursor()
            cursor.execute(sql_query)
            column_info = cursor.fetchall()
            df = pd.DataFrame(column_info, columns=["cid", "name", "type", "notnull", "default_value", "pk"])
            df = df[["name", "type"]]
            cursor.close()
            conn.close()
            return df
        else:
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            return df
            
    except Exception as e:
        error_msg = str(e)
        if "no such column" in error_msg:
            # Extract the problematic column name
            column_match = re.search(r"no such column: (\w+)", error_msg)
            if column_match:
                problematic_column = column_match.group(1)
                return f"Error: Column '{problematic_column}' not found in the data. Please check the available columns and try again."
        return f"Error executing SQL query: {error_msg}"

# Function to display columns when file is uploaded
def display_columns(csv_file):
    if csv_file is None:
        return "Please upload a CSV file first.", "No columns available"
    
    try:
        df = pd.read_csv(csv_file.name)
        cleaned_columns = [re.sub(r'[^\w]', '_', col).lower().strip('_') for col in df.columns]
        
        preview = df.head(3).to_string()
        column_info = f"Available columns ({len(cleaned_columns)}): {', '.join(cleaned_columns)}"
        
        return f"✅ CSV uploaded: {len(df)} rows, {len(df.columns)} columns", column_info
        
    except Exception as e:
        return f"❌ Error reading CSV file: {str(e)}", "No columns available"

# Enhanced main processing function
def process_query(csv_file, query, selected_model, use_ollama):
    if csv_file is None:
        return "Please upload a CSV file first.", "", pd.DataFrame()
    
    if not query.strip():
        return "Please enter a query.", "", pd.DataFrame()
    
    try:
        # Create database from CSV
        db_name, table_name, columns, original_columns = create_database_from_csv(csv_file.name)
        
        # Generate SQL query
        if use_ollama and check_ollama_available():
            sql_query = generate_sql_with_ollama(query, table_name, columns, selected_model)
            if sql_query is None:
                # Fallback to rule-based approach if Ollama fails
                sql_query = generate_sql_fallback(query, table_name, columns)
                status_msg = "⚠️ Ollama request failed/timed out, used fallback approach."
            else:
                status_msg = f"✅ Query generated using Ollama ({selected_model})."
        else:
            sql_query = generate_sql_fallback(query, table_name, columns)
            if use_ollama:
                status_msg = "⚠️ Ollama not available, used fallback approach."
            else:
                status_msg = "✅ Query generated using rule-based approach."
        
        # Execute SQL query
        result = execute_sql_query(db_name, sql_query)
        
        # Clean up the database file
        try:
            os.remove(db_name)
        except:
            pass
        
        return status_msg, sql_query, result
        
    except Exception as e:
        return f"❌ Error processing query: {str(e)}", "", pd.DataFrame()

# Function to update model dropdown
def refresh_models():
    models = get_available_models()
    if not models:
        return gr.Dropdown(choices=["No models available"], value="No models available")
    return gr.Dropdown(choices=models, value=models[0] if models else "")


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
                            height="600px",
                            object_fit="contain",
                            preview=True
                            )


                # Handle submission
                def on_submit(pdf_selection, query):
                    if not pdf_selection:
                        return "Please select a financial report first.", None, []

                    company = pdf_selection.split(":")[0].lower()
                    answer, doc_page_mapping, image_paths = process_query_and_generate(company, query)

                    # Load all cited page images using the proper helper
                    image_objects = []
                    for path in image_paths:
                        if os.path.exists(path):
                            try:
                                image_objects.append(Image.open(path))
                            except Exception as e:
                                print(f"Warning: Failed to open image {path}: {e}")

                    # Convert answer to speech using gTTS
                    tts = gTTS(answer)
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tts.save(temp_file.name)

                    return answer, temp_file.name, image_objects

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
                    model_selector = gr.Dropdown(choices=["phi3:mini", "mistral", "llama3", "gemma"], label="Ollama Model", value="phi3:mini")
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

            # === Tab 3: SQL Agent Tab ===
            with gr.Tab("📄 SQL Agent Tab"):
                gr.Markdown("# 🔍 Enhanced SQL RAG System with Ollama Integration")
                gr.Markdown("Upload a CSV file and ask questions about your data in natural language. Enhanced with better error handling and column matching.")
                
                with gr.Row():
                    csv_file = gr.File(label="📁 Upload CSV File", file_types=[".csv"])
                
                with gr.Row():
                    upload_status = gr.Textbox(label="📊 Upload Status", interactive=False)
                    columns_display = gr.Textbox(label="📋 Available Columns", lines=4, interactive=False)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        use_ollama = gr.Checkbox(label="🤖 Use Ollama for SQL Generation", value=ollama_available)
                    with gr.Column(scale=2):
                        model_dropdown = gr.Dropdown(
                            label="🔧 Select Ollama Model",
                            choices=get_available_models(),
                            value=get_available_models()[0] if get_available_models() else "",
                            interactive=True
                        )
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("🔄 Refresh Models", size="sm")
                
                with gr.Row():
                    query_input = gr.Textbox(
                        label="💬 Ask a question about your data",
                        placeholder="e.g., 'What is the average price?' or 'Show me the top 5 highest sales records'",
                        lines=2
                    )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 Submit Query", variant="primary", size="lg")
                
                with gr.Row():
                    status_output = gr.Textbox(label="📈 Query Status", interactive=False)
                
                with gr.Row():
                    sql_output = gr.Textbox(label="💾 Generated SQL Query", lines=2, interactive=False)
                
                with gr.Row():
                    result_output = gr.Dataframe(label="📊 Results", interactive=False)
                
                # Event handlers
                csv_file.change(
                    fn=display_columns,
                    inputs=[csv_file],
                    outputs=[upload_status, columns_display]
                )
                
                refresh_btn.click(
                    fn=refresh_models,
                    outputs=[model_dropdown]
                )
                
                submit_btn.click(
                    fn=process_query,
                    inputs=[csv_file, query_input, model_dropdown, use_ollama],
                    outputs=[status_output, sql_output, result_output]
                )


    return demo


# Launch the Gradio app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)