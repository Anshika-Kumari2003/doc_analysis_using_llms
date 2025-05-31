from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from youtube_search import YoutubeSearch
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
import requests
from llm_pipe.Ingestion_Retrieval.pinecone_retrieval import process_query, init_pinecone_and_embeddings

# Load environment variables
load_dotenv()

# Initialize Pinecone and embeddings
pc, index, embeddings_model = init_pinecone_and_embeddings()

# Configure Ollama API
OLLAMA_API_BASE = "http://localhost:11434/api"
OLLAMA_MODEL = "phi3:mini"

def query_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Helper function to query Ollama API"""
    try:
        response = requests.post(
            f"{OLLAMA_API_BASE}/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 512
                }
            },
            timeout=180
        )
        if response.status_code == 200:
            return response.json().get("response", "Error generating response.")
        else:
            return f"Error: Ollama returned status code {response.status_code}"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

def format_video_info(videos):
    base_url = "https://www.youtube.com"
    result = []

    for video in videos:
        url = f"{base_url}{video.get('url_suffix', '')}"
        channel = video.get('channel', '')
        title = video.get('title', 'No title')
        
        formatted_info = (
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Channel: {channel}\n"
        )
        
        result.append(formatted_info)
    
    return "\n".join(result)

@tool
def get_answer_enesys(question: str) -> str:
    """
    Get answers to questions about EnerSys from their 10-K filings.
    Use this tool when the question is specifically about EnerSys company, its financials, or operations.
    """
    try:
        results = process_query(index, embeddings_model, "enersys", question)
        if not results.get("results"):
            return "I couldn't find relevant information about EnerSys to answer your question. The documents may not contain the specific information you're looking for."
        
        # Format the context from results
        context = ""
        for doc_id, pages in results.get("results", {}).items():
            for page_num, matches in pages.items():
                for match in matches:
                    context += f"[Page {page_num}] {match.get('text', '').strip()}\n\n"
        
        if not context.strip():
            return "No relevant context found in EnerSys documents for your question."
        
        # Generate answer using the context
        prompt = f"""Based on the following context from EnerSys 10-K filings, answer the question correctly and concisely. 
        Do not answer in just one word. Provide a clear and concise answer.
Do not mention that you're using context or documents. Just provide the answer.
If the context doesn't contain the specific information needed, say that the information is not available in the documents.

Context:
{context}

Question: {question}

Answer:"""
        
        response = query_ollama(prompt)
        return response.strip()
        
    except Exception as e:
        return f"Error retrieving information about EnerSys: {str(e)}"

@tool
def get_answer_apple(question: str) -> str:
    """
    Get answers to questions about Apple from their 10-K filings.
    Use this tool when the question is specifically about Apple company, its financials, or operations.
    """
    try:
        results = process_query(index, embeddings_model, "apple", question)
        if not results.get("results"):
            return "I couldn't find relevant information about Apple to answer your question. The documents may not contain the specific information you're looking for."
        
        # Format the context from results
        context = ""
        for doc_id, pages in results.get("results", {}).items():
            for page_num, matches in pages.items():
                for match in matches:
                    context += f"[Page {page_num}] {match.get('text', '').strip()}\n\n"
        
        if not context.strip():
            return "No relevant context found in Apple documents for your question."
        
        # Generate answer using the context
        prompt = f"""Based on the following context from Apple 10-K filings, answer the question correctly and concisely.
        Do not answer in just one word. Provide a clear and concise answer.
Do not mention that you're using context or documents. Just provide the answer.
If the context doesn't contain the specific information needed, say that the information is not available in the documents.

Context:
{context}

Question: {question}

Answer:"""
        
        response = query_ollama(prompt)
        return response.strip()
        
    except Exception as e:
        return f"Error retrieving information about Apple: {str(e)}"

@tool
def get_answer_nvidia(question: str) -> str:
    """
    Get answers to questions about NVIDIA from their 10-K filings.
    Use this tool when the question is specifically about NVIDIA company, its financials, or operations.
    """
    try:
        results = process_query(index, embeddings_model, "nvidia", question)
        if not results.get("results"):
            return "I couldn't find relevant information about NVIDIA to answer your question. The documents may not contain the specific information you're looking for."
        
        # Format the context from results
        context = ""
        for doc_id, pages in results.get("results", {}).items():
            for page_num, matches in pages.items():
                for match in matches:
                    context += f"[Page {page_num}] {match.get('text', '').strip()}\n\n"
        
        if not context.strip():
            return "No relevant context found in NVIDIA documents for your question."
        
        # Generate answer using the context
        prompt = f"""Based on the following context from NVIDIA 10-K filings, answer the question correctly and concisely.
        Do not answer in just one word. Provide a clear and concise answer.
Do not mention that you're using context or documents. Just provide the answer.
If the context doesn't contain the specific information needed, say that the information is not available in the documents.

Context:
{context}

Question: {question}

Answer:"""
        
        response = query_ollama(prompt)
        return response.strip()
        
    except Exception as e:
        return f"Error retrieving information about NVIDIA: {str(e)}"

@tool
def youtube_url(question: str) -> str:
    """
    Search for relevant YouTube videos based on the question.
    Use this tool when the user might benefit from video content about the topic.
    """
    try:
        results = YoutubeSearch(question, max_results=4).to_dict()
        if not results:
            return "No relevant YouTube videos found for this topic."
        return format_video_info(results)
    except Exception as e:
        return f"Error searching YouTube: {str(e)}"

@tool
def web_search(question: str) -> str:
    """
    Search the web using Tavily API for additional context.
    Use this tool when the question requires current or external informations other than enersys, tesla , or nvidia.
    You can also use this tool when user asks about general things.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "Error: Tavily API key not found in environment variables."
    
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": tavily_api_key,
                "query": question,
                "search_depth": "basic",
                "include_answer": True,
                "include_domains": [],
                "exclude_domains": [],
                "max_results": 3 
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            results = []
            
            # Add the answer if available
            if "answer" in data and data["answer"]:
                results.append(f"Answer: {data['answer']}\n")
            
            # Add the top results
            for result in data.get("results", [])[:3]:
                results.append(f"Title: {result.get('title', 'N/A')}")
                results.append(f"URL: {result.get('url', 'N/A')}")
                results.append(f"Content: {result.get('content', 'N/A')}\n")
            
            return "\n".join(results) if results else "No relevant information found."
        else:
            return f"Error: Tavily API returned status code {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# List of available tools
agents = [
    get_answer_enesys,
    get_answer_apple,
    get_answer_nvidia,
    youtube_url,
    web_search
]