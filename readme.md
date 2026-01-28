# PDF to Markdown Converter using Mistral OCR API

A Python tool that converts PDF documents to Markdown format using Mistral AI's Document AI OCR processor (`mistral-ocr-latest` model).

---

## 📖 About Mistral OCR

**Mistral OCR** is a powerful Document AI service that extracts text and structured content from PDF documents with high accuracy.

### Key Features:
- Extracts text while maintaining document structure and hierarchy
- Preserves formatting like headers, paragraphs, lists, and tables
- Handles complex layouts including multi-column text and mixed content
- Returns results in Markdown format for easy parsing and rendering
- Detects and extracts hyperlinks

### Official Documentation:
- [Mistral OCR Documentation](https://docs.mistral.ai/capabilities/document_ai/basic_ocr)
- [Mistral API Reference](https://docs.mistral.ai/api/endpoint/ocr)

---

## ⚠️ IMPORTANT: API Key Security

> [!CAUTION]
> **Your API key is like a password - treat it with extreme care!**
> 
> - **NEVER** commit your API key to Git or share it publicly
> - **NEVER** share your API key with others
> - **NEVER** include your API key in code files
> - Use environment variables to store your API key
> - Regenerate your API key immediately if you suspect it has been exposed
> - API usage incurs costs - unauthorized access could result in unexpected charges

---

## 🚀 Features

- ✅ **Single file or folder processing** - Process individual PDFs or entire folders
- ✅ **Automatic PDF splitting** - Handles files > 50 MB or > 1000 pages by splitting them
- ✅ **Rate limiting** - Configurable sleep timer between API calls to avoid rate limits
- ✅ **Processing log** - Tracks processed files to avoid duplicate API calls (saves credits!)
- ✅ **Image preservation** - Optionally extract and save images from PDFs
- ✅ **Progress tracking** - Visual progress bar for batch processing
- ✅ **Double confirmation** - Confirms output path before processing

---

## 📋 Prerequisites

- Python 3.10 or higher
- A Mistral AI API key (get one at [console.mistral.ai](https://console.mistral.ai))

---

## 🛠️ Installation

### Step 1: Clone or Download the Repository

```bash
git clone <repository-url>
cd mistral
```

### Step 2: Create a Python Virtual Environment

Using `uv` (recommended):
```bash
uv venv mistral_pyenv --python 3.13
source mistral_pyenv/bin/activate  # On macOS/Linux
# OR
mistral_pyenv\Scripts\activate     # On Windows
```

Or using standard Python:
```bash
python -m venv mistral_pyenv
source mistral_pyenv/bin/activate  # On macOS/Linux
# OR
mistral_pyenv\Scripts\activate     # On Windows
```

### Step 3: Install Dependencies

Using `uv`:
```bash
uv pip install -r requirements.txt
```

Or using `pip`:
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Your API Key

**On macOS/Linux:**
```bash
export MISTRAL_API_KEY='your-api-key-here'
```

**On Windows (Command Prompt):**
```cmd
set MISTRAL_API_KEY=your-api-key-here
```

**On Windows (PowerShell):**
```powershell
$env:MISTRAL_API_KEY="your-api-key-here"
```

> 💡 **Tip:** Add the export command to your shell profile (`.bashrc`, `.zshrc`, etc.) to make it permanent.

---

## 📖 Usage

### Running the Script

```bash
python pdf_to_markdown.py
```

### Interactive Prompts

The script will guide you through the process:

1. **Input Path**: Enter a PDF file path OR a folder containing PDFs
2. **Output Directory**: Specify where to save the Markdown files
3. **Confirm Output Path**: Double confirmation to prevent accidental overwrites
4. **Preserve Images**: Choose whether to extract and save images (yes/no)
5. **Sleep Timer**: Set delay between API calls (default: 1 second)

### Example Session

```
============================================================
PDF to Markdown Converter (Mistral OCR)
============================================================

You can provide either:
  - A single PDF file path
  - A folder path (to process all PDFs in it)

Enter the input path: /path/to/your/pdfs/

📁 FOLDER INPUT DETECTED
Found 5 PDF file(s) in the folder:
  • document1.pdf (2.34 MB)
  • document2.pdf (15.67 MB)
  • large_book.pdf (120.45 MB) (will be split: >50MB, >1000 pages)

Do you want to process all these files? (yes/no): yes

Enter the output directory path: ./output

============================================================
OUTPUT PATH CONFIRMATION
============================================================
Output will be saved to: ./output

Confirm this output path? (yes/no): yes
Please confirm again to proceed (yes/no): yes

Preserve images from PDFs? (yes/no, default: no): yes
  ✓ Images will be saved to: ./output/images/

Enter sleep time between API calls in seconds (default: 1): 2

Initializing Mistral client...

============================================================
STARTING PROCESSING
============================================================

Overall Progress: [██████░░░░░░░░░░░░░░░░░░░░░░░░] 1/5 (20.0%)
Processing: document1.pdf (2.34 MB)
  📄 Processing file...
  🖼️  Extracting images...
  ✅ Saved 12 image(s)
  ✅ Saved: document1.md (45 pages)

...

============================================================
CONVERSION COMPLETE!
============================================================
Files processed: 5
Total pages: 892
Total images saved: 156
Output directory: ./output
Processing log: ./output/pdf_processing_log.json
```

---

## 📁 Output Structure

```
output/
├── document1.md
├── document2.md
├── large_book_part1.md
├── large_book_part2.md
├── pdf_processing_log.json
└── images/
    ├── document1/
    │   ├── page1_img-0.jpeg
    │   └── page2_img-1.png
    └── document2/
        └── page1_img-0.jpeg
```

---

## 📝 Processing Log

The script maintains a `pdf_processing_log.json` file that tracks:
- Which files have been processed
- Output file paths
- Number of pages processed
- Number of images saved
- Processing timestamp

**This prevents re-processing the same files and saves API credits!**

---

## ⚙️ API Limits

| Limit | Value |
|-------|-------|
| Maximum file size | 50 MB per request |
| Maximum pages | 1000 pages per request |

Files exceeding these limits are automatically split into smaller chunks.

---

## 🔧 Troubleshooting

### "MISTRAL_API_KEY environment variable not set"
Make sure you've set the environment variable correctly:
```bash
export MISTRAL_API_KEY='your-api-key-here'
```

### "PermissionError: Operation not permitted"
On macOS, you may need to grant Terminal access to the folder:
1. Go to **System Preferences** → **Privacy & Security** → **Full Disk Access**
2. Add **Terminal** (or your terminal app)

### Rate Limiting Errors
Increase the sleep timer between API calls:
```
Enter sleep time between API calls in seconds (default: 1): 5
```

---

## 📄 License

MIT License - feel free to use and modify as needed.

---

## 🔗 Resources

- [Mistral AI Official Website](https://mistral.ai)
- [Mistral AI Console](https://console.mistral.ai)
- [Mistral OCR Documentation](https://docs.mistral.ai/capabilities/document_ai/basic_ocr)
- [Mistral API Reference](https://docs.mistral.ai/api/endpoint/ocr)