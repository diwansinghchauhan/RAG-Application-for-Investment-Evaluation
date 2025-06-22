# RAG-based Investment Evaluation App

This application enables investors to upload a startup pitch deck (PDF) and ask meaningful questions using a Retrieval-Augmented Generation (RAG) pipeline powered by local LLMs like Mistral (via Ollama). It extracts text and OCR content, builds a searchable FAISS index using sentence-transformer embeddings and queries that index to generate context-aware answers.

## Demo

[https://github.com/user-attachments/assets/28d7b196-37a2-449f-8ddb-58621fd47d1c](https://github.com/user-attachments/assets/28d7b196-37a2-449f-8ddb-58621fd47d1c)

## Features

- Upload and process pitch decks (PDF)
- Extract both text and images with OCR
- Build FAISS vector index for semantic search
- Query top relevant text chunks using SentenceTransformers
- Generate answers using local Mistral model through Ollama
- Lightweight, fast, and works offline

## Approach

**Text Extraction**

- Extracts text from PDFs using `PyMuPDF` for structured content.
- Uses **TrOCR** (VisionEncoderDecoder) to perform OCR on embedded images or scanned content.

**Chunking**

- Extracted content is split into smaller `Document` chunks for better semantic embedding and retrieval.

**Embedding and Indexing**

- Chunks are embedded using `sentence-transformers/paraphrase-MiniLM-L6-v2`.
- Embedded vectors are indexed using FAISS for fast similarity search.

**Querying**

- A user query is semantically matched against the indexed chunks.
- Top-K relevant contexts are retrieved.

**Answer Generation**

- The relevant context is sent to the Mistral model using `ollama run mistral` and the answer is returned.

## Tech Stack

| Category          | Technology / Library                                                            | Purpose                                                    |
| ----------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **PDF Parsing**   | [`PyMuPDF (fitz)`](https://pymupdf.readthedocs.io/)                             | Extracts structured text and embedded images from PDFs     |
| **OCR**           | [`TrOCR`](https://huggingface.co/microsoft/trocr-base-printed) + `transformers` | Extracts text from images using vision-to-text OCR         |
| **Embeddings**    | [`sentence-transformers`](https://www.sbert.net/)                               | Converts text into dense semantic vector embeddings        |
| **Vector Search** | [`FAISS`](https://github.com/facebookresearch/faiss)                            | Enables fast similarity search across embedded text chunks |
| **LLM Inference** | [`Ollama`](https://ollama.com) + Mistral (`mistral`)                            | Generates answers using local language models              |
| **App Framework** | [`Streamlit`](https://streamlit.io)                                             | Web interface for uploading PDFs and asking questions      |
| **Core Language** | `Python 3.9`                                                                    | Primary programming language                               |
| **Optional GPU**  | `PyTorch`                                                                       | Accelerates TrOCR if CUDA is available                     |

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/diwansinghchauhan/RAG-Application-for-Investment-Evaluation.git
cd RAG-Application-for-Investment-Evaluation
```

### 2. Create & Activate a Virtual Environment

```bash
# For macOS/Linux
python -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Run Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Start the Mistral model
ollama run mistral
```

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

Then open your browser and go to:\
[http://localhost:8501](http://localhost:8501)

## 🔗 Links

&#x20;

