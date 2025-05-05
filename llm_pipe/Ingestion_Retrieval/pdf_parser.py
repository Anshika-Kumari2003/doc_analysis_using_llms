import os
import re
import pandas as pd
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.layout import LAParams
import tabula
import json

def parse_pdf(file_path):
    """
    Parse PDF file to extract text with preserved formatting and tables.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        dict: Dictionary containing:
            - 'text': Extracted text with preserved formatting
            - 'tables': List of tables as strings in a tabular format
            - 'combined_content': Text and tables combined for chunking
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")
    
    # Extract text with formatting preserved
    laparams = LAParams(all_texts=True)
    text = pdfminer_extract_text(file_path, laparams=laparams)

    
    # Clean up text: normalize line endings and remove excessive whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)  # Limit consecutive newlines to 2
    
    # Extract tables
    tables_data = []
    tables_text = []
    tables_with_context = []
    
    try:
        # Extract all tables from the PDF
        tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
        
        # Convert tables to text representation for chunking
        for i, table in enumerate(tables):
            tables_data.append(table)
            
            # Convert table to string with clear formatting
            table_str = f"\n--- TABLE {i+1} ---\n"
            table_str += table.to_string(index=False)
            table_str += "\n--- END TABLE ---\n"
            
            tables_text.append(table_str)
            
            # Store table as JSON for structured retrieval
            table_json = table.to_json(orient="records")
            table_dict = json.loads(table_json)
            
            # Create a semantic description of the table
            try:
                table_name = f"Table {i+1}"
                columns = table.columns.tolist()
                row_count = len(table)
                
                # Extract context (text before the table)
                # Find position in text where this table might be referenced
                table_context = extract_table_context(text, table, columns)
                
                tables_with_context.append({
                    "table_id": i+1,
                    "table_data": table_dict,
                    "column_names": columns,
                    "row_count": row_count,
                    "table_context": table_context,
                    "table_description": f"{table_name}: A table with {row_count} rows and {len(columns)} columns. Columns include: {', '.join(str(col) for col in columns)}"
                })
            except Exception as e:
                print(f"Warning: Error processing table metadata: {e}")
            
    except Exception as e:
        print(f"Warning: Error extracting tables: {e}")
    
    # Create combined content with text and tables
    combined_content = text
    
    # Add tables with their context for better semantic understanding
    if tables_with_context:
        for table_info in tables_with_context:
            table_section = f"\n\n--- TABLE {table_info['table_id']} ---\n"
            if table_info['table_context']:
                table_section += f"Context: {table_info['table_context']}\n"
            table_section += f"Description: {table_info['table_description']}\n"
            table_section += "Data:\n"
            
            # Format table data in a query-friendly way
            formatted_data = format_table_for_queries(table_info['table_data'], table_info['column_names'])
            table_section += formatted_data
            table_section += "\n--- END TABLE ---\n"
            
            combined_content += f"\n\n{table_section}"
    
    return {
        "text": text,
        "tables": tables_data,
        "tables_text": tables_text,
        "tables_with_context": tables_with_context,
        "combined_content": combined_content
    }

def extract_table_context(text, table, columns):
    """
    Attempt to extract contextual information about a table from surrounding text.
    """
    # Find potential table references or headers in the text
    # Look for key column names or table titles in the text
    context = ""
    
    # Convert column names to a regex pattern
    if columns and len(columns) > 0:
        # Get the first few column names for matching
        sample_columns = [str(col) for col in columns[:min(3, len(columns))]]
        
        # Create regex patterns to find text that might reference the table
        for col in sample_columns:
            if len(col) > 3:  # Avoid very short column names
                # Find sentences that mention this column name
                pattern = f"[^.!?]*{re.escape(col)}[^.!?]*[.!?]"
                matches = re.findall(pattern, text)
                if matches:
                    context += " ".join(matches[:2]) + " "
    
    # Look for "Table X" references
    table_refs = re.findall(r"Table\s+\d+[^.!?]*[.!?]", text)
    if table_refs:
        context += " ".join(table_refs[:2])
    
    return context.strip()

def format_table_for_queries(table_data, column_names):
    """
    Format table data in a way that's optimized for query answering.
    """
    if not table_data:
        return ""
    
    # Create a formatted string representation
    formatted = ""
    
    # Add header
    formatted += " | ".join(str(col) for col in column_names) + "\n"
    formatted += "-" * 50 + "\n"
    
    # Add rows
    for row in table_data:
        row_values = [str(row.get(col, "")) for col in column_names]
        formatted += " | ".join(row_values) + "\n"
    
    return formatted

def extract_content_for_embedding(file_path):
    """
    Extract content from PDF file ready for chunking and embedding.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Combined text content with tables, ready for chunking
    """
    if not file_path.endswith('.pdf'):
        raise ValueError("Unsupported file type. Only PDF files are supported.")
    
    result = parse_pdf(file_path)
    return result["combined_content"]

#Example usage
if __name__ == "__main__":
    pdf_path = "sample1.pdf"
    
    # Get detailed parsing results
    parsed_content = parse_pdf(pdf_path)
    
    print(f"Text extraction successful: {len(parsed_content['text']) > 0}")
    print(f"Number of tables extracted: {len(parsed_content['tables'])}")
    
    # Get content ready for chunking and embedding
    content_for_embedding = extract_content_for_embedding(pdf_path)
    print(f"Total content length: {len(content_for_embedding)} characters")
    
    # Print sample of the content
    print("\nSample of content ready for chunking:")
    print(content_for_embedding)