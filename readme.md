# Open-Sourced RAG Pipeline
![RAG Workflow](assets/RAG.png "RAG Workflow")
Transform your document processing workflow with this powerful Retrieval-Augmented Generation (RAG) pipeline. This project provides a robust solution for extracting insights from various document formats, including PDFs, images, and text files, using state-of-the-art language models and vector embeddings.

## Project Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- Sufficient disk space for model storage and document processing
- Memory: Minimum 16GB RAM (32GB recommended for larger models)

## Dependencies

The project relies on several key libraries:

- Transformers and AutoGPTQ for language model inference
- ChromaDB for vector storage and retrieval
- LangChain for text processing and chunking
- Various OCR and document processing libraries

A complete list of dependencies can be found in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Getting Started

1. Ensure you have the required dependencies installed
2. Prepare your documents in one of the supported formats:
   - PDF files
   - Images (PNG, JPG, JPEG)
   - Text files (TXT)

## How to Run the Application

The pipeline can be run using the `rag_pipeline.py` script with the following command:

```bash
python rag_pipeline.py -s path/to/your/document -q "Your query here"
```

For example, to process a PDF document and ask a question:

```bash
python rag_pipeline.py -s document.pdf -q "What are the key responsibilities mentioned in this document?"
```

### Supported Models

The pipeline supports several language models:
- OpenChat 3.5
- Mixtral 8x7B
- Vicuna 13B v1.5 16K
- Zephyr 7B Beta

## Relevant Code Examples

### Document Processing

The pipeline automatically handles different document types:

```python
# For images
text = text_from_pytess("image.jpg")

# For PDFs
conv_pdf_to_img("document.pdf", "temp/img")
text = process_pdf_images("temp/")

# For text files
with open("document.txt", "r") as file:
    text = file.read()
```

### Vector Storage and Retrieval

The system uses ChromaDB for efficient vector storage and retrieval:

```python
# Initialize vector store
client = chromadb.PersistentClient(path="/path/to/db")
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# Store document chunks
collection.add(
    embeddings=vectors,
    documents=chunks,
    ids=ids
)
```

## Conclusion

This RAG pipeline provides a powerful and flexible solution for document processing and information retrieval. Whether you're working with PDFs, images, or text files, the system can help you extract valuable insights efficiently.

Key features:
- Support for multiple document formats
- State-of-the-art language models
- Efficient vector storage and retrieval
- Flexible querying capabilities

We welcome contributions and feedback to improve the project. Feel free to open issues or submit pull requests!
