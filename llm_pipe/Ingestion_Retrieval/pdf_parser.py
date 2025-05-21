import os
import warnings
import pdfplumber
import pandas as pd

# Suppress UserWarnings from pdfplumber (like CropBox missing warnings)
warnings.filterwarnings("ignore", category=UserWarning, module='pdfplumber')

def is_table_row(line):
    tokens = line.split()
    numeric_count = sum(1 for t in tokens if t.replace(',', '').replace('.', '').isdigit())
    return numeric_count >= 3 and len(tokens) >= 4

def convert_table_to_markdown(table_data):
    if not table_data or len(table_data) < 2:
        return ""

    df = pd.DataFrame(table_data[1:], columns=table_data[0])
    df.columns = df.columns.astype(str)

    markdown = "| " + " | ".join(df.columns) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
    for _, row in df.iterrows():
        markdown += "| " + " | ".join(str(cell).strip() for cell in row) + " |\n"
    return markdown

def parse_pdf(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    combined_text = ""

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total pages to process: {total_pages}")

        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
                text = text.strip()

                lines = text.splitlines()
                clean_lines = [line for line in lines if not is_table_row(line)]
                page_text_cleaned = "\n".join(clean_lines)

                combined_text += f"\n\n## Page {page_number}\n\n"
                combined_text += page_text_cleaned + "\n"

                tables = page.extract_tables()
                for idx, table in enumerate(tables):
                    markdown = convert_table_to_markdown(table)
                    if markdown:
                        combined_text += f"\n### Table {idx + 1} (Page {page_number})\n{markdown}\n"

            except Exception as e:
                print(f"Warning: Skipping page {page_number} due to error: {e}")

    return combined_text

# Example usage
# if __name__ == "__main__":
#     input_pdf = "test.pdf"
#     result = parse_pdf(input_pdf)
#     print(result)