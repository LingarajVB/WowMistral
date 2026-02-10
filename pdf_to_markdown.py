#!/usr/bin/env python3
"""
PDF to Markdown Converter using Mistral OCR API

This script converts PDF documents to Markdown format using Mistral's
Document AI OCR processor (mistral-ocr-latest model).

Features:
    - Single file or folder processing
    - Automatic PDF splitting for files >= 50 MB
    - Rate limiting with configurable sleep timer
    - Processing log to avoid duplicate API calls
    - Double confirmation for output paths
    - Optional image preservation
    - Progress tracking

Requirements:
    pip install mistralai PyPDF2

Usage:
    python pdf_to_markdown.py

Environment Variables:
    MISTRAL_API_KEY: Your Mistral API key (required)
"""

import os
import re
import sys
import json
import time
import base64
import tempfile
from pathlib import Path
from datetime import datetime

try:
    from mistralai import Mistral
except ImportError:
    print("Error: mistralai package not installed.")
    print("Please install it using: pip install mistralai")
    sys.exit(1)

try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    print("Error: PyPDF2 package not installed.")
    print("Please install it using: pip install PyPDF2")
    sys.exit(1)

# Check for pydantic (needed for image annotation)
try:
    from pydantic import BaseModel, Field
    from enum import Enum
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# Image annotation schema (only if pydantic is available)
if PYDANTIC_AVAILABLE:
    from mistralai.extra import response_format_from_pydantic_model
    
    class ImageCategory(str, Enum):
        """Classification categories for image content."""
        NEUTRAL = "neutral"
        POSITIVE = "positive"
        NEGATIVE = "negative"
        INFORMATIVE = "informative"
    
    class ImageAnnotation(BaseModel):
        """Structured annotation schema for extracted images."""
        category: ImageCategory = Field(..., description="Classification category of the image content: neutral, positive, negative, or informative")
        confidence: float = Field(..., description="Confidence score between 0.0 and 1.0 indicating how confident the model is in this classification")
        reasoning: str = Field(..., description="Detailed explanation describing the image content and why this classification category was chosen")


# Maximum file size in bytes (50 MB)
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Maximum pages per API request (Mistral limit is 1000)
MAX_PAGES_PER_REQUEST = 1000

# Log file name
PROCESSING_LOG_FILE = "pdf_processing_log.json"

# Images subfolder name
IMAGES_FOLDER = "images"


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def get_file_size_bytes(file_path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)


def load_processing_log(output_dir: str) -> dict:
    """Load the processing log from the output directory."""
    log_path = os.path.join(output_dir, PROCESSING_LOG_FILE)
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"processed_files": {}}
    return {"processed_files": {}}


def save_processing_log(output_dir: str, log_data: dict) -> None:
    """Save the processing log to the output directory."""
    log_path = os.path.join(output_dir, PROCESSING_LOG_FILE)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)


def is_already_processed(log_data: dict, pdf_path: str, output_path: str) -> bool:
    """Check if a PDF has already been processed and output exists."""
    pdf_name = os.path.basename(pdf_path)
    if pdf_name in log_data.get("processed_files", {}):
        entry = log_data["processed_files"][pdf_name]
        # Check if all output files exist
        output_files = entry.get("output_files", [])
        if output_files and all(os.path.exists(f) for f in output_files):
            return True
    return False


def log_processed_file(log_data: dict, pdf_path: str, output_files: list, pages: int, images_saved: int = 0) -> None:
    """Log a processed file."""
    pdf_name = os.path.basename(pdf_path)
    log_data["processed_files"][pdf_name] = {
        "source_path": pdf_path,
        "output_files": output_files,
        "pages_processed": pages,
        "images_saved": images_saved,
        "processed_at": datetime.now().isoformat(),
        "file_size_mb": get_file_size_mb(pdf_path)
    }


def validate_path(file_path: str) -> tuple[bool, bool, str]:
    """
    Validate the input path (file or directory).
    
    Returns:
        tuple: (is_valid, is_directory, error_message)
    """
    path = Path(file_path)
    
    # Check if path exists
    if not path.exists():
        return False, False, f"Path not found: {file_path}"
    
    # Check if it's a directory
    if path.is_dir():
        return True, True, ""
    
    # Check if it's a file with PDF extension
    if path.is_file():
        if path.suffix.lower() != '.pdf':
            return False, False, f"File is not a PDF: {file_path} (extension: {path.suffix})"
        return True, False, ""
    
    return False, False, f"Invalid path: {file_path}"


def get_pdf_files_from_folder(folder_path: str) -> list[str]:
    """Get all PDF files from a folder."""
    pdf_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(folder_path, file))
    return sorted(pdf_files)


def get_pdf_page_count(pdf_path: str) -> int:
    """Get the number of pages in a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except:
        return 0


def check_if_needs_split(pdf_path: str) -> tuple[bool, str]:
    """
    Check if a PDF needs to be split.
    
    Returns:
        tuple: (needs_split, reason_string)
    """
    size_mb = get_file_size_mb(pdf_path)
    page_count = get_pdf_page_count(pdf_path)
    
    reasons = []
    if size_mb >= MAX_FILE_SIZE_MB:
        reasons.append(f">{MAX_FILE_SIZE_MB}MB")
    if page_count > MAX_PAGES_PER_REQUEST:
        reasons.append(f">{MAX_PAGES_PER_REQUEST} pages")
    
    if reasons:
        return True, f" (will be split: {', '.join(reasons)})"
    return False, ""

def split_pdf_if_needed(pdf_path: str, max_size_bytes: int = MAX_FILE_SIZE_BYTES, max_pages: int = MAX_PAGES_PER_REQUEST) -> list[str]:
    """
    Split a PDF into smaller chunks based on file size OR page count.
    
    Splits if:
    - File size >= max_size_bytes (default 50 MB)
    - Page count > max_pages (default 1000 pages, Mistral API limit)
    
    Returns a list of paths to the split PDF files (temporary files).
    If the file is already under both limits, returns a list with just the original path.
    """
    file_size = get_file_size_bytes(pdf_path)
    
    # Read PDF to get page count
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    # Check if splitting is needed
    needs_size_split = file_size >= max_size_bytes
    needs_page_split = total_pages > max_pages
    
    if not needs_size_split and not needs_page_split:
        return [pdf_path]
    
    # Determine reason for split
    split_reasons = []
    if needs_size_split:
        split_reasons.append(f"{get_file_size_mb(pdf_path):.2f} MB > {MAX_FILE_SIZE_MB} MB")
    if needs_page_split:
        split_reasons.append(f"{total_pages} pages > {max_pages} page limit")
    
    print(f"  PDF needs splitting ({', '.join(split_reasons)})...")
    
    # Calculate pages per chunk considering both limits
    avg_page_size = file_size / total_pages if total_pages > 0 else 0
    
    # Pages per chunk based on size limit (use 90% to be safe)
    if avg_page_size > 0:
        pages_by_size = max(1, int((max_size_bytes * 0.9) / avg_page_size))
    else:
        pages_by_size = max_pages
    
    # Pages per chunk based on page limit (use 90% to be safe)
    pages_by_count = int(max_pages * 0.9)
    
    # Use the smaller of the two limits
    pages_per_chunk = min(pages_by_size, pages_by_count)
    
    chunks = []
    chunk_num = 1
    temp_dir = tempfile.mkdtemp(prefix="pdf_split_")
    
    print(f"    Splitting into chunks of ~{pages_per_chunk} pages each...")
    
    for start_page in range(0, total_pages, pages_per_chunk):
        end_page = min(start_page + pages_per_chunk, total_pages)
        
        writer = PdfWriter()
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])
        
        # Write chunk to temp file
        base_name = Path(pdf_path).stem
        chunk_path = os.path.join(temp_dir, f"{base_name}_part{chunk_num}.pdf")
        
        with open(chunk_path, 'wb') as f:
            writer.write(f)
        
        chunks.append(chunk_path)
        print(f"    Created chunk {chunk_num}: pages {start_page + 1}-{end_page}")
        chunk_num += 1
    
    return chunks


def encode_pdf_to_base64(file_path: str) -> str:
    """Read and encode a PDF file to base64."""
    with open(file_path, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode('utf-8')


def process_pdf_with_ocr(client: Mistral, pdf_path: str, include_images: bool = True, annotate_images: bool = False) -> dict:
    """
    Process a PDF file using Mistral OCR API.
    
    Args:
        client: Mistral client instance
        pdf_path: Path to the PDF file
        include_images: Whether to include image data in the response
        annotate_images: Whether to annotate images with classification
        
    Returns:
        dict: OCR response containing extracted content
    """
    # Encode PDF to base64
    pdf_base64 = encode_pdf_to_base64(pdf_path)
    
    # Create data URI for the PDF
    document_url = f"data:application/pdf;base64,{pdf_base64}"
    
    # Build OCR parameters
    ocr_params = {
        "model": "mistral-ocr-latest",
        "document": {
            "type": "document_url",
            "document_url": document_url
        },
        "include_image_base64": include_images
    }
    
    # Add image annotation if requested and pydantic is available
    if annotate_images and include_images and PYDANTIC_AVAILABLE:
        ocr_params["bbox_annotation_format"] = response_format_from_pydantic_model(ImageAnnotation)
    
    # Process with OCR
    ocr_response = client.ocr.process(**ocr_params)
    
    return ocr_response


def save_images_from_response(ocr_response, output_dir: str, pdf_base_name: str) -> tuple[dict, dict]:
    """
    Save images from OCR response to files.
    
    Args:
        ocr_response: OCR API response object
        output_dir: Directory to save images
        pdf_base_name: Base name of the PDF file (used for image naming)
        
    Returns:
        tuple: (image_mapping dict, annotations dict)
            - image_mapping: Mapping of original image references to saved file paths
            - annotations: Mapping of image IDs to their annotation data
    """
    image_mapping = {}
    annotations = {}
    images_dir = os.path.join(output_dir, IMAGES_FOLDER, pdf_base_name)
    
    # Create images directory
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    
    for page in ocr_response.pages:
        if not hasattr(page, 'images') or not page.images:
            continue
            
        for img in page.images:
            # Get image ID and base64 data
            img_id = getattr(img, 'id', None)
            img_base64 = getattr(img, 'image_base64', None)
            
            if not img_id or not img_base64:
                continue
            
            # Get annotation if available
            img_annotation = getattr(img, 'image_annotation', None)
            if img_annotation:
                try:
                    # Parse JSON annotation string
                    if isinstance(img_annotation, str):
                        annotations[img_id] = json.loads(img_annotation)
                    else:
                        annotations[img_id] = img_annotation
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Determine file extension from the image ID or default to png
            # Image IDs are typically like "img-0.jpeg", "img-1.png", etc.
            if '.' in img_id:
                ext = img_id.split('.')[-1].lower()
            else:
                ext = 'png'
            
            # Clean the image ID for use as filename
            safe_img_id = re.sub(r'[^\w\-.]', '_', img_id)
            img_filename = f"page{page.index + 1}_{safe_img_id}"
            if not img_filename.endswith(f'.{ext}'):
                img_filename = f"{img_filename}.{ext}"
            
            img_path = os.path.join(images_dir, img_filename)
            
            try:
                # Decode and save image
                # Handle data URI format if present
                if ',' in img_base64:
                    img_base64 = img_base64.split(',')[1]
                
                img_data = base64.b64decode(img_base64)
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                
                # Create relative path for markdown
                relative_path = os.path.join(IMAGES_FOLDER, pdf_base_name, img_filename)
                image_mapping[img_id] = relative_path
                
            except Exception as e:
                print(f"    ⚠️  Failed to save image {img_id}: {e}")
    
    return image_mapping, annotations


def update_markdown_with_images(markdown: str, image_mapping: dict, annotations: dict = None) -> str:
    """
    Update markdown content to reference saved image files.
    Uses angle brackets around paths to handle filenames with spaces.
    
    Args:
        markdown: Original markdown content
        image_mapping: Mapping of original image references to saved file paths
        annotations: Optional mapping of image IDs to annotation data
        
    Returns:
        str: Updated markdown with correct image paths and optional annotations
    """
    updated_markdown = markdown
    
    for original_ref, new_path in image_mapping.items():
        # Build annotation block if available
        annotation_block = ""
        if annotations and original_ref in annotations:
            ann = annotations[original_ref]
            category = ann.get('category', 'unknown')
            confidence = ann.get('confidence', 0)
            reasoning = ann.get('reasoning', 'No description available')
            annotation_block = f"\n\n> **Image Analysis:**\n> - Category: {category}\n> - Confidence: {confidence}\n> - Reasoning: {reasoning}"
        
        # Replace image references like ![img-0.jpeg](img-0.jpeg)
        # with ![img-0.jpeg](<images/pdf_name/page1_img-0.jpeg>) using angle brackets
        pattern = rf'!\[([^\]]*)\]\({re.escape(original_ref)}\)'
        replacement = f'![\\1](<{new_path}>){annotation_block}'
        updated_markdown = re.sub(pattern, replacement, updated_markdown)
        
        # Also handle cases where just the filename is referenced
        pattern2 = rf'\]\({re.escape(original_ref)}\)'
        replacement2 = f'](<{new_path}>)'
        updated_markdown = re.sub(pattern2, replacement2, updated_markdown)
    
    return updated_markdown


def extract_markdown_from_response(ocr_response, image_mapping: dict = None, annotations: dict = None) -> str:
    """
    Extract and combine markdown content from all pages.
    
    Args:
        ocr_response: OCR API response object
        image_mapping: Optional mapping of image references to file paths
        annotations: Optional mapping of image IDs to annotation data
        
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
        page_markdown = page.markdown
        
        # Update image references if mapping provided
        if image_mapping:
            page_markdown = update_markdown_with_images(page_markdown, image_mapping, annotations)
        
        markdown_parts.append(page_markdown)
    
    return "".join(markdown_parts)


def save_markdown(content: str, output_path: str) -> None:
    """Save markdown content to a file."""
    # Create parent directories if they don't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


def get_output_path_for_pdf(pdf_path: str, output_dir: str, part_num: int = None) -> str:
    """Generate output path for a PDF file."""
    base_name = Path(pdf_path).stem
    
    # Remove _partN suffix if present (for split files)
    if "_part" in base_name:
        # Keep the part suffix from the split file
        pass
    elif part_num is not None:
        base_name = f"{base_name}_part{part_num}"
    
    return os.path.join(output_dir, f"{base_name}.md")


def double_confirm_output_path(output_path: str) -> bool:
    """Ask for double confirmation of the output path."""
    print(f"\n{'='*60}")
    print("OUTPUT PATH CONFIRMATION")
    print(f"{'='*60}")
    print(f"Output will be saved to: {output_path}")
    
    confirm1 = input("\nConfirm this output path? (yes/no): ").strip().lower()
    if confirm1 not in ['yes', 'y']:
        return False
    
    confirm2 = input("Please confirm again to proceed (yes/no): ").strip().lower()
    if confirm2 not in ['yes', 'y']:
        return False
    
    return True


def print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    """Print a progress indicator."""
    percentage = (current / total) * 100 if total > 0 else 0
    bar_length = 30
    filled = int(bar_length * current / total) if total > 0 else 0
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\n{prefix}: [{bar}] {current}/{total} ({percentage:.1f}%)")


def process_single_pdf(
    client: Mistral,
    pdf_path: str,
    output_dir: str,
    log_data: dict,
    sleep_seconds: float,
    preserve_images: bool = False,
    annotate_images: bool = False,
    current_file: int = 1,
    total_files: int = 1
) -> tuple[list[str], int, int]:
    """
    Process a single PDF file, handling splitting if needed.
    
    Returns:
        tuple: (list of output file paths, total pages processed, images saved)
    """
    pdf_name = os.path.basename(pdf_path)
    size_mb = get_file_size_mb(pdf_path)
    
    print_progress(current_file, total_files, "Overall Progress")
    print(f"Processing: {pdf_name} ({size_mb:.2f} MB)")
    
    # Check if already processed
    base_output = get_output_path_for_pdf(pdf_path, output_dir)
    if is_already_processed(log_data, pdf_path, base_output):
        print(f"  ⏭️  Already processed (found in log). Skipping...")
        entry = log_data["processed_files"][pdf_name]
        return entry.get("output_files", []), entry.get("pages_processed", 0), entry.get("images_saved", 0)
    
    # Split PDF if needed
    try:
        pdf_chunks = split_pdf_if_needed(pdf_path)
    except Exception as e:
        print(f"  ❌ Error reading PDF (may be corrupted or use unsupported format): {e}")
        print(f"  ⏭️  Skipping this file...")
        return [], 0, 0
    
    is_split = len(pdf_chunks) > 1
    
    output_files = []
    total_pages = 0
    total_images = 0
    
    for i, chunk_path in enumerate(pdf_chunks):
        chunk_num = i + 1 if is_split else None
        
        # Determine output path
        if is_split:
            # Use the chunk filename which already has _partN
            output_path = get_output_path_for_pdf(chunk_path, output_dir)
            pdf_base_name = Path(chunk_path).stem
        else:
            output_path = get_output_path_for_pdf(pdf_path, output_dir)
            pdf_base_name = Path(pdf_path).stem
        
        # Check if this specific output already exists
        if os.path.exists(output_path):
            print(f"  ⏭️  Output already exists: {os.path.basename(output_path)}. Skipping...")
            output_files.append(output_path)
            continue
        
        chunk_info = f"chunk {chunk_num}/{len(pdf_chunks)}" if is_split else "file"
        print(f"  📄 Processing {chunk_info}...")
        
        try:
            # Process with OCR
            ocr_response = process_pdf_with_ocr(client, chunk_path, include_images=preserve_images, annotate_images=annotate_images)
            
            # Save images if requested
            image_mapping = {}
            annotations = {}
            images_saved = 0
            if preserve_images:
                print(f"  🖼️  Extracting images...")
                image_mapping, annotations = save_images_from_response(ocr_response, output_dir, pdf_base_name)
                images_saved = len(image_mapping)
                if images_saved > 0:
                    print(f"  ✅ Saved {images_saved} image(s)")
                    if annotations:
                        print(f"  📝 Generated {len(annotations)} annotation(s)")
                else:
                    print(f"  ℹ️  No images found in this document")
            
            # Extract markdown
            markdown_content = extract_markdown_from_response(ocr_response, image_mapping, annotations)
            
            # Save output
            save_markdown(markdown_content, output_path)
            output_files.append(output_path)
            total_pages += len(ocr_response.pages)
            total_images += images_saved
            
            print(f"  ✅ Saved: {os.path.basename(output_path)} ({len(ocr_response.pages)} pages)")
            
            # Sleep to avoid rate limiting (don't sleep after the last chunk)
            if sleep_seconds > 0 and (i < len(pdf_chunks) - 1):
                print(f"  ⏳ Sleeping {sleep_seconds}s to avoid rate limiting...")
                time.sleep(sleep_seconds)
                
        except Exception as e:
            print(f"  ❌ Error processing: {e}")
            continue
    
    # Clean up temporary split files
    if is_split:
        for chunk_path in pdf_chunks:
            if os.path.exists(chunk_path) and chunk_path != pdf_path:
                try:
                    os.remove(chunk_path)
                except:
                    pass
    
    # Log the processed file
    if output_files:
        log_processed_file(log_data, pdf_path, output_files, total_pages, total_images)
    
    return output_files, total_pages, total_images


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
    
    # Get input path
    print("You can provide either:")
    print("  - A single PDF file path")
    print("  - A folder path (to process all PDFs in it)")
    print()
    input_path = input("Enter the input path: ").strip()
    
    # Remove quotes if present (common when dragging files)
    input_path = input_path.strip('"').strip("'")
    
    # Expand user home directory if present
    input_path = os.path.expanduser(input_path)
    
    # Validate input path
    is_valid, is_directory, error_msg = validate_path(input_path)
    if not is_valid:
        print(f"\nError: {error_msg}")
        sys.exit(1)
    
    # Get list of PDF files to process
    if is_directory:
        pdf_files = get_pdf_files_from_folder(input_path)
        if not pdf_files:
            print(f"\nNo PDF files found in folder: {input_path}")
            sys.exit(1)
        
        print(f"\n📁 FOLDER INPUT DETECTED")
        print(f"Found {len(pdf_files)} PDF file(s) in the folder:")
        total_size = 0
        for pdf in pdf_files:
            size = get_file_size_mb(pdf)
            total_size += size
            _, needs_split = check_if_needs_split(pdf)
            print(f"  • {os.path.basename(pdf)} ({size:.2f} MB){needs_split}")
        print(f"\nTotal size: {total_size:.2f} MB")
        
        proceed = input("\nDo you want to process all these files? (yes/no): ").strip().lower()
        if proceed not in ['yes', 'y']:
            print("Operation cancelled.")
            sys.exit(0)
    else:
        pdf_files = [input_path]
        size_mb = get_file_size_mb(input_path)
        _, needs_split = check_if_needs_split(input_path)
        print(f"\n📄 Single file: {size_mb:.2f} MB{needs_split}")
    
    # Get output directory
    print()
    output_path = input("Enter the output directory path: ").strip()
    
    # Remove quotes if present
    output_path = output_path.strip('"').strip("'")
    
    # Expand user home directory if present
    output_path = os.path.expanduser(output_path)
    
    # Ensure output path is a directory
    if os.path.exists(output_path) and not os.path.isdir(output_path):
        print(f"\nError: Output path exists but is not a directory: {output_path}")
        sys.exit(1)
    
    # Double confirmation for output path
    if not double_confirm_output_path(output_path):
        print("Operation cancelled.")
        sys.exit(0)
    
    # Create output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Ask about image preservation
    print()
    preserve_images_input = input("Preserve images from PDFs? (yes/no, default: no): ").strip().lower()
    preserve_images = preserve_images_input in ['yes', 'y']
    if preserve_images:
        print(f"  ✓ Images will be saved to: {os.path.join(output_path, IMAGES_FOLDER)}/")
    
    # Ask about image annotation (only if preserving images and pydantic is available)
    annotate_images = False
    if preserve_images:
        if PYDANTIC_AVAILABLE:
            print()
            annotate_input = input("Annotate images with AI classification? (yes/no, default: no): ").strip().lower()
            annotate_images = annotate_input in ['yes', 'y']
            if annotate_images:
                print("  ✓ Images will be annotated with category, confidence, and reasoning")
        else:
            print("  ℹ️  Image annotation requires pydantic. Install with: pip install pydantic")
    
    # Get sleep timer
    print()
    sleep_input = input("Enter sleep time between API calls in seconds (default: 1): ").strip()
    try:
        sleep_seconds = float(sleep_input) if sleep_input else 1.0
    except ValueError:
        print("Invalid number, using default of 1 second.")
        sleep_seconds = 1.0
    
    # Load processing log
    log_data = load_processing_log(output_path)
    
    # Check for already processed files
    already_processed = []
    to_process = []
    for pdf in pdf_files:
        base_output = get_output_path_for_pdf(pdf, output_path)
        if is_already_processed(log_data, pdf, base_output):
            already_processed.append(pdf)
        else:
            to_process.append(pdf)
    
    if already_processed:
        print(f"\n⏭️  {len(already_processed)} file(s) already processed (will be skipped):")
        for pdf in already_processed:
            print(f"  • {os.path.basename(pdf)}")
    
    if not to_process:
        print("\n✅ All files have already been processed!")
        sys.exit(0)
    
    print(f"\n📝 {len(to_process)} file(s) to process:")
    for pdf in to_process:
        print(f"  • {os.path.basename(pdf)}")
    
    # Initialize Mistral client
    print("\nInitializing Mistral client...")
    client = Mistral(api_key=api_key)
    
    # Process all PDFs
    print("\n" + "=" * 60)
    print("STARTING PROCESSING")
    print("=" * 60)
    
    total_files_processed = 0
    total_pages_processed = 0
    total_images_saved = 0
    all_output_files = []
    
    for i, pdf_path in enumerate(pdf_files):
        output_files, pages, images = process_single_pdf(
            client, pdf_path, output_path, log_data, sleep_seconds,
            preserve_images=preserve_images,
            annotate_images=annotate_images,
            current_file=i + 1,
            total_files=len(pdf_files)
        )
        
        if output_files:
            total_files_processed += 1
            total_pages_processed += pages
            total_images_saved += images
            all_output_files.extend(output_files)
        
        # Save log after each file
        save_processing_log(output_path, log_data)
        
        # Sleep between files (not after the last one)
        if sleep_seconds > 0 and i < len(pdf_files) - 1 and pdf_path in to_process:
            print(f"\n⏳ Sleeping {sleep_seconds}s before next file...")
            time.sleep(sleep_seconds)
    
    # Print summary
    print()
    print("=" * 60)
    print("CONVERSION COMPLETE!")
    print("=" * 60)
    print(f"Files processed: {total_files_processed}")
    print(f"Total pages: {total_pages_processed}")
    if preserve_images:
        print(f"Total images saved: {total_images_saved}")
    print(f"Output directory: {output_path}")
    print(f"Processing log: {os.path.join(output_path, PROCESSING_LOG_FILE)}")
    print()
    print("Output files:")
    for f in all_output_files:
        print(f"  • {os.path.basename(f)}")
    print()


if __name__ == "__main__":
    main()
