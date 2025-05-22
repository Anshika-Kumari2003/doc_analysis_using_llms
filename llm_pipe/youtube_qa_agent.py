import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi
import requests
import threading
import re
import json
import os

# Path for persistent cache
CACHE_FILE = "transcript_cache.json"
video_context = {}

# Load from disk if exists
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        video_context = json.load(f)

# Save cache
def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(video_context, f)

# Extract video ID
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

# Background transcript fetching
def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry['text'] for entry in transcript])
        video_context[video_id] = full_text
        save_cache()
    except Exception as e:
        video_context[video_id] = f"Error fetching transcript: {e}"
        save_cache()

# Handle URL submission
def handle_url_submit(url):
    video_id = extract_video_id(url)
    if not video_id:
        return "Invalid YouTube URL."
    if video_id not in video_context:
        threading.Thread(target=fetch_transcript, args=(video_id,)).start()
        return f"Transcript is being fetched in the background for video ID: {video_id}..."
    return "Transcript already cached. Ready for QA."

# Send a prompt to Ollama using /generate endpoint (consistent with main app)
def query_ollama(prompt, model="phi3:mini"):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 512
        }
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}"
    except KeyError:
        return "Error: Unexpected response format from Ollama"

# Answer a question
def answer_question(question, url, model):
    video_id = extract_video_id(url)
    if video_id not in video_context:
        return "Transcript not ready yet. Please fetch transcript first."
    context = video_context[video_id]
    if context.startswith("Error"):
        return context
    
    # Limit context to avoid token limits
    context_limited = context[:4000] if len(context) > 4000 else context
    
    prompt = f"""Based on the following YouTube video transcript, please answer the question accurately and concisely.

Transcript:
{context_limited}

Question: {question}

Answer:"""
    
    try:
        return query_ollama(prompt, model)
    except Exception as e:
        return f"Error from Ollama: {e}"

# Summarize transcript
def summarize_transcript(url, model):
    video_id = extract_video_id(url)
    if video_id not in video_context:
        return "Transcript not ready. Please fetch transcript first."
    context = video_context[video_id]
    if context.startswith("Error"):
        return context
    
    # Limit context to avoid token limits
    context_limited = context[:4000] if len(context) > 4000 else context
    
    prompt = f"""Please provide a comprehensive summary of the following YouTube video transcript. Include the main topics, key points, and important details discussed.

Transcript:
{context_limited}

Summary:"""
    
    try:
        return query_ollama(prompt, model)
    except Exception as e:
        return f"Error from Ollama: {e}"