# Multi-Modal RAG System

## 🎯 System Overview

This is a comprehensive Multi-Modal Retrieval-Augmented Generation (RAG) system that combines document question-answering, YouTube video analysis, and CSV data querying capabilities. The system leverages open-source models and provides an intuitive web interface for users to interact with various data sources.

### Key Capabilities
- **Document QA**: Process PDF files (SEC filings, financial reports) and answer questions with page citations
- **YouTube Analysis**: Extract transcripts from YouTube videos and provide Q&A functionality
- **SQL Agent**: Upload CSV files and query data using natural language
- **Multi-modal Output**: Text responses with visual citations and audio playback


## 🏗️ Architecture

The system follows a modular architecture with three main components:

<img src="Flow diagram/rag_system_flowchart.svg" alt="Architecture">


## 🔧 Core Components

### 1. Vector Database Integration
- **Pinecone**: Primary vector database for document embeddings
- **Sentence Transformers**: Generate embeddings for text chunks
- **Retrieval System**: Semantic search with relevance scoring

### 2. Language Model Integration
- **Ollama**: Local LLM server (primarily Phi-3 mini model)
- **Fallback Mechanisms**: Rule-based responses when LLM is unavailable
- **Model Management**: Dynamic model selection and availability checking

### 3. Data Processing Pipeline
- **PDF Processing**: Extract text and images with page mapping
- **CSV Processing**: Automatic schema detection and SQL generation
- **YouTube Processing**: Transcript extraction and analysis

### 4. Web Interface
- **Gradio Framework**: Multi-tab interface for different functionalities
- **Real-time Processing**: Async handling of queries and responses
- **Visual Feedback**: Progress indicators and status messages


## ✨ Features

### Document QA System
- 📄 **PDF Processing**: Support for financial reports and SEC filings
- 🔍 **Semantic Search**: Find relevant information across documents
- 📊 **Page Citations**: Automatic page number referencing in answers
- 🖼️ **Visual Citations**: Display source pages as images
- 🔊 **Audio Output**: Text-to-speech for accessibility

### YouTube QA Agent
- 🎥 **Video Analysis**: Extract and process YouTube video transcripts
- 💬 **Interactive Q&A**: Ask questions about video content
- 📝 **Summarization**: Generate concise video summaries
- 🔄 **Model Selection**: Choose from multiple Ollama models

### SQL Agent
- 📊 **CSV Upload**: Process CSV files with automatic schema detection
- 🧠 **Natural Language to SQL**: Convert questions to SQL queries
- 🔧 **Column Matching**: Fuzzy matching for column name recognition
- 📈 **Result Visualization**: Display query results in tabular format


## 🔄 System Workflow

### Document QA Workflow
```
User Upload PDF → Process & Chunk → Generate Embeddings → Store in Pinecone
                                                               ↓
User Query → Semantic Search → Retrieve Relevant Chunks → Generate Answer → Display with Citations
```

### YouTube QA Workflow
```
YouTube URL → Extract Transcript → Store Locally → Process with Ollama → Generate Responses
```

### SQL Agent Workflow
```
CSV Upload → Create SQLite DB → Natural Language Query → Generate SQL → Execute → Return Results
```


## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Ollama installed and running locally
- Pinecone account and API key

### Required Models
```bash
# Install Ollama models
ollama pull phi3:mini
ollama pull mistral
ollama pull llama3
ollama pull gemma
```

### Python Dependencies
```bash
pip install gradio
pip install pinecone-client
pip install sentence-transformers
pip install pandas
pip install sqlite3
pip install gtts
pip install youtube-transcript-api
pip install requests
pip install python-dotenv
pip install Pillow
```


## ⚙️ Configuration

### Environment Variables
Create a `.env` file with:
```env
PINECONE_API_KEY=your_pinecone_api_key
INDEX_NAME=your_index_name
MODELS_DIR=./models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_MODEL=phi3:mini
```

### Company PDF Mapping
Configure document mapping in `config.py`:
```python
COMPANY_PDF_MAPPING = {
    "apple": ["apple_10k_2023.pdf", "apple_annual_report.pdf"],
    "microsoft": ["msft_earnings.pdf"],
    # Add more mappings as needed
}
```


## 📖 Usage Guide

### Document QA System
1. **Select Document**: Choose from available financial reports
2. **Enter Query**: Ask specific questions about the document
3. **Get Results**: Receive answers with page citations and visual references
4. **Audio Playback**: Listen to the generated response

### YouTube QA Agent
1. **Enter URL**: Paste YouTube video URL
2. **Fetch Transcript**: Extract video transcript
3. **Ask Questions**: Query about video content
4. **Summarize**: Generate video summary

### SQL Agent
1. **Upload CSV**: Select and upload CSV file
2. **View Schema**: Review available columns
3. **Natural Query**: Ask questions in plain English
4. **View Results**: Examine generated SQL and results


## 🔌 API Endpoints

### Ollama Integration
- **Health Check**: `GET /api/tags` - Check available models
- **Generate**: `POST /api/generate` - Generate text responses
- **Models**: `GET /api/tags` - List available models

### Internal Functions
- `process_query_and_generate()` - Main document QA processing
- `handle_url_submit()` - YouTube transcript extraction
- `process_query_sql()` - CSV query processing


## 📁 File Structure

```
passion-project-doc-analysis-using-llms-25-bhupender-a/
├── .gradio/
├── jsons/
├── llm_pipe/
│   ├── Ingestion_Retrieval/
│   │   ├── __init__.py
│   │   ├── document_chunker.py
│   │   ├── pdf_parser.py
│   │   ├── pinecone_integration.py
│   │   └── pinecone_retrieval.py
│   ├── gradio_app.py
│   ├── main.py
│   └── youtube_qa_agent.py
├── page_images/
├── pdfs/
├── reports/
├── .DS_Store
├── .env
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── config.py
├── docker-compose.yml
├── full-requirements.txt
├── requirements.txt
└── vertex_key.json
```


## 📦 Dependencies

### Core Libraries
- **gradio**: Web interface framework
- **pinecone-client**: Vector database operations
- **sentence-transformers**: Text embeddings
- **pandas**: Data manipulation
- **requests**: HTTP client for API calls

### AI/ML Libraries
- **transformers**: Hugging Face transformers
- **torch**: PyTorch for model operations
- **numpy**: Numerical computations

### Utility Libraries
- **python-dotenv**: Environment variable management
- **Pillow**: Image processing
- **gtts**: Text-to-speech conversion
- **sqlite3**: Database operations


## 🐛 Troubleshooting

### Common Issues

#### Ollama Connection Issues
```python
# Check Ollama status
def check_ollama_available():
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=5)
        return response.status_code == 200
    except:
        return False
```

**Solution**: Ensure Ollama is running on `localhost:11434`

#### Pinecone Authentication
```
Error: Pinecone API key not found
```
**Solution**: Verify `.env` file contains valid `PINECONE_API_KEY`

#### CSV Encoding Issues
```
UnicodeDecodeError: 'utf-8' codec can't decode
```
**Solution**: System automatically tries multiple encodings (latin-1, cp1252, iso-8859-1)

#### Memory Issues
```
CUDA out of memory
```
**Solution**: Use CPU-based models or reduce batch size

### Performance Optimization

#### Vector Database
- Use appropriate index dimensions
- Implement batch processing for large documents
- Regular index maintenance

#### Model Selection
- Use `phi3:mini` for faster responses
- Use `llama3` for better accuracy
- Monitor GPU/CPU usage

#### Query Optimization
- Implement query caching
- Use connection pooling for databases
- Optimize chunk sizes for retrieval

### Monitoring & Logging

#### System Health Checks
```python
# Monitor system components
def system_health_check():
    checks = {
        'ollama': check_ollama_available(),
        'pinecone': check_pinecone_connection(),
        'models': check_model_availability()
    }
    return checks
```

#### Performance Metrics
- Query response times
- Model loading times
- Vector similarity scores
- Database query performance


## 🚀 Future Enhancements

### Planned Features
- **Multi-language Support**: Support for non-English documents
- **Advanced Analytics**: Query performance analytics
- **Batch Processing**: Handle multiple files simultaneously
- **API Integration**: RESTful API for external applications
- **User Management**: Authentication and user sessions

### Technical Improvements
- **Caching Layer**: Redis integration for faster responses
- **Model Fine-tuning**: Domain-specific model adaptations
- **Advanced Retrieval**: Hybrid search combining semantic and keyword search
- **Scalability**: Kubernetes deployment support

---

*This documentation provides a comprehensive overview of the Multi-Modal RAG system. For specific implementation details, refer to the source code and configuration files.*
