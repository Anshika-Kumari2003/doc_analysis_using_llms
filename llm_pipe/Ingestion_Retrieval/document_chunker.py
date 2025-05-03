import uuid
import re

# LangChain imports
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from langchain_core.documents import Document


def create_text_splitter(chunk_size=500, chunk_overlap=100, strategy="recursive"):
    """
    Create a text splitter based on the specified strategy.
    """
    if strategy == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    elif strategy == "character":
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif strategy == "token":
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")


def chunk_documents(docs, chunk_size=500, chunk_overlap=100, strategy="recursive"):
    """
    Split documents into smaller chunks with metadata.
    Tables are automatically detected and processed separately.
    """
    # Extract tables first to process them separately
    regular_docs = []
    table_sections = []
    
    for doc in docs:
        # Process the document content
        content = doc.page_content
        
        # Identify table sections using regex
        table_pattern = r'(--- TABLE \d+ ---.*?--- END TABLE ---)'
        matches = re.finditer(table_pattern, content, re.DOTALL)
        
        # Extract table sections and their positions
        positions = []
        for match in matches:
            table_sections.append(Document(
                page_content=match.group(1),
                metadata={
                    **doc.metadata,
                    "content_type": "table",
                    "is_table": True
                }
            ))
            positions.append((match.start(), match.end()))
        
        # Remove table sections from original text
        if positions:
            new_content = ""
            last_end = 0
            
            for start, end in sorted(positions):
                new_content += content[last_end:start]
                last_end = end
                
            new_content += content[last_end:]
            
            # Only add non-table content if there's something left
            if new_content.strip():
                regular_docs.append(Document(
                    page_content=new_content,
                    metadata={
                        **doc.metadata,
                        "content_type": "text",
                        "is_table": False
                    }
                ))
        else:
            # No tables found, keep original document
            regular_docs.append(Document(
                page_content=content,
                metadata={
                    **doc.metadata,
                    "content_type": "text",
                    "is_table": False
                }
            ))
    
    # Process regular text with standard chunking
    text_splitter = create_text_splitter(chunk_size, chunk_overlap, strategy)
    chunked_text_docs = text_splitter.split_documents(regular_docs)
    
    # Add metadata to text chunks
    for i, doc in enumerate(chunked_text_docs):
        doc.metadata.update({
            "chunk_id": str(uuid.uuid4()),
            "chunk_index": i,
            "total_chunks": len(chunked_text_docs),
            "chunking_strategy": strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "page": doc.metadata.get("page", "unknown"),
            "source": doc.metadata.get("source", "unknown"),
            "doc_id": doc.metadata.get("doc_id", str(uuid.uuid4())),
        })
    
    # Process table sections with specialized chunking
    chunked_table_docs = chunk_tables_intelligently(table_sections)
    
    # Combine both types of chunks
    return chunked_text_docs + chunked_table_docs


def chunk_tables_intelligently(table_docs):
    """
    Chunk table data more intelligently to preserve context and structure.
    """
    chunked_tables = []
    
    for i, doc in enumerate(table_docs):
        content = doc.page_content
        
        # Extract table ID, context, description and data sections
        table_id_match = re.search(r'--- TABLE (\d+) ---', content)
        table_id = table_id_match.group(1) if table_id_match else f"unknown-{i}"
        
        context_match = re.search(r'Context: (.*?)(?:\n|$)', content)
        context = context_match.group(1) if context_match else ""
        
        description_match = re.search(r'Description: (.*?)(?:\n|$)', content)
        description = description_match.group(1) if description_match else ""
        
        # Extract the actual table data
        data_match = re.search(r'Data:\n(.*?)(?:\n--- END TABLE ---|$)', content, re.DOTALL)
        table_data = data_match.group(1) if data_match else ""
        
        # Split the table into header and rows
        table_lines = table_data.strip().split('\n')
        header = table_lines[0] if table_lines else ""
        divider = table_lines[1] if len(table_lines) > 1 else ""
        data_rows = table_lines[2:] if len(table_lines) > 2 else []
        
        # Create a preamble with context and description
        preamble = f"--- TABLE {table_id} ---\n"
        if context:
            preamble += f"Context: {context}\n"
        preamble += f"Description: {description}\n"
        
        # Create chunks for the table
        # Always include the header in each chunk
        max_rows_per_chunk = 30  # Adjust based on your needs
        
        if len(data_rows) <= max_rows_per_chunk:
            # Small table - keep it as one chunk
            full_content = f"{preamble}Data:\n{header}\n{divider}\n" + "\n".join(data_rows) + "\n--- END TABLE ---"
            chunked_tables.append(Document(
                page_content=full_content,
                metadata={
                    **doc.metadata,
                    "chunk_id": str(uuid.uuid4()),
                    "is_table": True,
                    "table_id": table_id,
                    "table_chunk": "complete",
                    "row_range": f"1-{len(data_rows)}",
                    "doc_id": doc.metadata.get("doc_id", str(uuid.uuid4())),
                }
            ))
        else:
            # Large table - chunk it by rows
            for j in range(0, len(data_rows), max_rows_per_chunk):
                end_idx = min(j + max_rows_per_chunk, len(data_rows))
                chunk_rows = data_rows[j:end_idx]
                
                chunk_content = f"{preamble}Data:\n{header}\n{divider}\n" + "\n".join(chunk_rows) + "\n--- END TABLE ---"
                chunked_tables.append(Document(
                    page_content=chunk_content,
                    metadata={
                        **doc.metadata,
                        "chunk_id": str(uuid.uuid4()),
                        "is_table": True,
                        "table_id": table_id,
                        "table_chunk": f"part_{j//max_rows_per_chunk+1}_of_{(len(data_rows) + max_rows_per_chunk - 1) // max_rows_per_chunk}",
                        "row_range": f"{j+1}-{end_idx}",
                        "doc_id": doc.metadata.get("doc_id", str(uuid.uuid4())),
                    }
                ))
    
    return chunked_tables


def process_documents(docs, chunk_size=500, chunk_overlap=100, strategy="recursive"):
    """
    Main processing function for the chunking pipeline.
    
    Tables are automatically detected and processed separately from regular text.
    The function returns a combined list of text chunks and table chunks.
    
    Args:
        docs: List of Document objects to process
        chunk_size: Size of chunks for regular text
        chunk_overlap: Overlap between chunks for regular text
        strategy: Chunking strategy for regular text
        
    Returns:
        List of Document objects with chunks of both text and tables
    """
    return chunk_documents(docs, chunk_size, chunk_overlap, strategy)


# Example usage for testing
# if __name__ == "__main__":
#     from pdf_parser import extract_content_for_embedding, parse_pdf
    
#     # Same sample file used in pdf_parser.py
#     pdf_path = "sample1.pdf"
    
#     # Get content ready for chunking from the parser
#     content = extract_content_for_embedding(pdf_path)
    
#     # Create a Document object
#     doc = Document(
#         page_content=content,
#         metadata={"source": pdf_path}
#     )
    
#     # Process the document through our chunking pipeline
#     chunks = process_documents(
#         [doc],
#         chunk_size=500,
#         chunk_overlap=100
#     )
    
#     # Print statistics about the chunks
#     total_chunks = len(chunks)
#     table_chunks = sum(1 for chunk in chunks if chunk.metadata.get("is_table", False))
#     text_chunks = total_chunks - table_chunks
    
#     print(f"\nChunking Results for {pdf_path}:")
#     print(f"Total chunks created: {total_chunks}")
#     print(f"Text chunks: {text_chunks}")
#     print(f"Table chunks: {table_chunks}")
    
#     # Print all text chunks
#     print("\n==== TEXT CHUNKS ====")
#     for i, chunk in enumerate(chunks):
#         if not chunk.metadata.get("is_table", False):
#             print(f"\n--- TEXT CHUNK {i+1} ---")
#             print(f"Chunk ID: {chunk.metadata.get('chunk_id')}")
#             print(f"Content type: {chunk.metadata.get('content_type')}")
#             print(f"Content:")
#             print(chunk.page_content)
#             print("-" * 50)
    
#     # Print all table chunks
#     print("\n==== TABLE CHUNKS ====")
#     for i, chunk in enumerate(chunks):
#         if chunk.metadata.get("is_table", False):
#             print(f"\n--- TABLE CHUNK {i+1} ---")
#             print(f"Chunk ID: {chunk.metadata.get('chunk_id')}")
#             print(f"Table ID: {chunk.metadata.get('table_id')}")
#             print(f"Table chunk: {chunk.metadata.get('table_chunk')}")
#             print(f"Row range: {chunk.metadata.get('row_range')}")
#             print(f"Content:")
#             print(chunk.page_content)
#             print("-" * 50)