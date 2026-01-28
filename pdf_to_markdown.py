#!/usr/bin/env python3
"""
PDF to Markdown Converter using Mistral OCR API

This script converts PDF documents to Markdown format using Mistral's
Document AI OCR processor (mistral-ocr-latest model).

Requirements:
    pip install mistralai

Usage:
    python pdf_to_markdown.py

Environment Variables:
    MISTRAL_API_KEY: Your Mistral API key (required)
"""

import os
import sys
import base64
from pathlib import Path

try:
    from mistralai import Mistral
except ImportError:
    print("Error: mistralai package not installed.")
    print("Please install it using: pip install mistralai")
    sys.exit(1)


# Maximum file size in bytes (50 MB)
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def validate_pdf_file(file_path: str) -> tuple[bool, str]:
    """
    Validate the input PDF file.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        return False, f"File not found: {file_path}"
    
    # Check if it's a file (not a directory)
    if not path.is_file():
        return False, f"Path is not a file: {file_path}"
    
    # Check file extension
    if path.suffix.lower() != '.pdf':
        return False, f"File is not a PDF: {file_path} (extension: {path.suffix})"
    
    # Check file size
    size_mb = get_file_size_mb(file_path)
    if size_mb >= MAX_FILE_SIZE_MB:
        return False, (
            f"File size ({size_mb:.2f} MB) exceeds the maximum allowed size "
            f"of {MAX_FILE_SIZE_MB} MB. Please use a smaller file."
        )
    
    return True, ""


def encode_pdf_to_base64(file_path: str) -> str:
    """Read and encode a PDF file to base64."""
    with open(file_path, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode('utf-8')


def process_pdf_with_ocr(client: Mistral, pdf_path: str) -> dict:
    """
    Process a PDF file using Mistral OCR API.
    
    Args:
        client: Mistral client instance
        pdf_path: Path to the PDF file
        
    Returns:
        dict: OCR response containing extracted content
    """
    # Encode PDF to base64
    pdf_base64 = encode_pdf_to_base64(pdf_path)
    
    # Create data URI for the PDF
    document_url = f"data:application/pdf;base64,{pdf_base64}"
    
    # Process with OCR
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": document_url
        },
        include_image_base64=True
    )
    
    return ocr_response


def extract_markdown_from_response(ocr_response) -> str:
    """
    Extract and combine markdown content from all pages.
    
    Args:
        ocr_response: OCR API response object
        
    Returns:
        str: Combined markdown content from all pages
    """
    markdown_parts = []
    
    for page in ocr_response.pages:
        # Add page separator for multi-page documents
        if page.index > 0:
            markdown_parts.append(f"\n\n---\n\n<!-- Page {page.index + 1} -->\n\n")
        else:
            markdown_parts.append(f"<!-- Page {page.index + 1} -->\n\n")
        
        # Add the markdown content
        markdown_parts.append(page.markdown)
    
    return "".join(markdown_parts)


def save_markdown(content: str, output_path: str) -> None:
    """Save markdown content to a file."""
    # Create parent directories if they don't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


def main():
    """Main function to run the PDF to Markdown converter."""
    print("=" * 60)
    print("PDF to Markdown Converter (Mistral OCR)")
    print("=" * 60)
    print()
    
    # Check for API key
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set.")
        print("Please set it using: export MISTRAL_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Get input PDF path
    print(f"Note: Maximum file size allowed is {MAX_FILE_SIZE_MB} MB")
    print()
    input_path = input("Enter the input PDF file path: ").strip()
    
    # Remove quotes if present (common when dragging files)
    input_path = input_path.strip('"').strip("'")
    
    # Expand user home directory if present
    input_path = os.path.expanduser(input_path)
    
    # Validate input file
    is_valid, error_msg = validate_pdf_file(input_path)
    if not is_valid:
        print(f"\nError: {error_msg}")
        sys.exit(1)
    
    # Show file info
    size_mb = get_file_size_mb(input_path)
    print(f"\nFile size: {size_mb:.2f} MB ✓")
    
    # Get output path
    print()
    output_path = input("Enter the output Markdown file path: ").strip()
    
    # Remove quotes if present
    output_path = output_path.strip('"').strip("'")
    
    # Expand user home directory if present
    output_path = os.path.expanduser(output_path)
    
    # Ensure output has .md extension
    if not output_path.lower().endswith('.md'):
        output_path += '.md'
    
    # Check if output file already exists
    if os.path.exists(output_path):
        overwrite = input(f"\nFile '{output_path}' already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Operation cancelled.")
            sys.exit(0)
    
    # Initialize Mistral client
    print("\nInitializing Mistral client...")
    client = Mistral(api_key=api_key)
    
    # Process the PDF
    print(f"Processing PDF with OCR (this may take a moment)...")
    try:
        ocr_response = process_pdf_with_ocr(client, input_path)
    except Exception as e:
        print(f"\nError during OCR processing: {e}")
        sys.exit(1)
    
    # Extract markdown
    print("Extracting markdown content...")
    markdown_content = extract_markdown_from_response(ocr_response)
    
    # Save to file
    print(f"Saving to: {output_path}")
    save_markdown(markdown_content, output_path)
    
    # Print summary
    print()
    print("=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Pages processed: {len(ocr_response.pages)}")
    print()


if __name__ == "__main__":
    main()
