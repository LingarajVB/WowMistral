# Chat Conversation

Note: _This is purely the output of the chat conversation and does not contain any raw data, codebase snippets, etc. used to generate the output._

### User Input

https://docs.mistral.ai/capabilities/document_ai/basic_ocr

Can you go through this properly and then tell me what's the best way to do this kind of a conversion locally in the sense that like like uh uh heating hitting the API through the Python script. 
The script should basically ask for an input PDF file path and it should give me the output PDF file path and when I input the PDF file path it should also check the size. If the size is greater than or equal to 50 MB then it should not process it. 

*Read URL content from /capabilities/document_ai/basic_ocr *

*Listed directory [mistral](file:///Users/lingarajbv/terminal/ai_ws/github/mistral) *

### Planner Response

Let me read the relevant chunks from the documentation to understand the OCR API:

*Viewed content chunk*

*Viewed content chunk*

*Viewed content chunk*

*Viewed content chunk*

*Viewed content chunk*

### Planner Response

Let me also check the base64 encoding method for local files:

*Viewed content chunk*

*Viewed content chunk*

*Viewed content chunk*

*Viewed content chunk*

### Planner Response

Now I have a good understanding of the Mistral OCR API. Let me create the Python script for you:

## Summary of Mistral OCR API

Based on the documentation, here's what you need to know:

### Key Points:
1. **Model**: Use `mistral-ocr-latest`
2. **API Endpoint**: `client.ocr.process()`
3. **Input Methods for PDFs**:
   - **Public URL**: Pass a document URL directly
   - **Base64 encoded**: For local files, encode them as base64
   - **Upload to Mistral Cloud**: Upload first, then reference

4. **Output**: Returns markdown format with:
   - Page-by-page extracted text (`pages[].markdown`)
   - Images and tables as placeholders that can be mapped
   - Hyperlinks detected
   - Optional header/footer extraction

### For Local Files (Your Use Case):
Since you want to process local PDFs, the best approach is to:
1. Read the local PDF file
2. Base64 encode it  
3. Send it to the API with `type: "document_url"` using a data URI

Let me create the script for you:

*Edited relevant file*

### Planner Response

I've created the script for you. Here's what I built:

## PDF to Markdown Converter using Mistral OCR

### How It Works

The script uses Mistral's **Document AI OCR processor** (`mistral-ocr-latest` model) to convert PDF files to Markdown:

1. **Input Validation**: Checks if the file exists, is a valid PDF, and is under 50 MB
2. **Base64 Encoding**: Reads the local PDF and encodes it as base64 (this is how you send local files to the API)
3. **API Call**: Sends the encoded PDF to `client.ocr.process()` 
4. **Output**: Extracts the markdown content from each page and combines them into a single output file

### Setup

1. **Install the Mistral SDK**:
   ```bash
   pip install mistralai
   ```

2. **Set your API key**:
   ```bash
   export MISTRAL_API_KEY='your-api-key-here'
   ```

3. **Run the script**:
   ```bash
   python /Users/lingarajbv/terminal/ai_ws/github/mistral/pdf_to_markdown.py
   ```

### Features

- ✅ Prompts for input PDF path and output Markdown path
- ✅ Validates file size (rejects files ≥ 50 MB)
- ✅ Handles multi-page PDFs with page separators
- ✅ Strips quotes from paths (useful when drag-and-drop)
- ✅ Expands `~` home directory shorthand
- ✅ Asks before overwriting existing files
- ✅ Shows progress and summary

### Key API Details (from the docs)

For local files, the best approach is to use **base64 encoding** with a data URI:
```python
document_url = f"data:application/pdf;base64,{pdf_base64}"

ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": document_url
    }
)
```

The response contains pages with a [markdown](cci:1://file:///Users/lingarajbv/terminal/ai_ws/github/mistral/pdf_to_markdown.py:134:0-140:24) field containing the extracted text. Would you like me to run a test or make any modifications to the script?