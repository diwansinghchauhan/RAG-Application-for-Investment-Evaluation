from langchain.schema import Document
import fitz
import pytesseract
from PIL import Image
import io
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import subprocess

# 1. Text and OCR Extraction
def extract_text_and_images(file_path):
    raw_text = extract_text(file_path)
    text_chunks = [Document(page_content=raw_text[i:i+500]) for i in range(0, len(raw_text), 500)]

    doc = fitz.open(file_path)
    ocr_docs = []
    for page_index in range(len(doc)):
        for img_index, img in enumerate(doc[page_index].get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            ocr_text = pytesseract.image_to_string(img_pil)
            if ocr_text.strip():
                ocr_docs.append(Document(
                    page_content=ocr_text,
                    metadata={"source": f"img_pg{page_index+1}_img_{img_index+1}"}
                ))

    return text_chunks + ocr_docs

# 2. FAISS Indexing
def build_faiss_index(documents):
    texts = [doc.page_content for doc in documents]
    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(384)
    index.add(np.array(embeddings).astype("float32"))
    return index, texts, model

# 3. Query FAISS
def query_faiss(query, model, index, texts, k=3):
    q_vec = model.encode([query]).astype("float32")
    D, I = index.search(q_vec, k)
    return [texts[i] for i in I[0]]

# 4. Call Mistral via Ollama
def ask_mistral(context, question, model="mistral"):
    prompt = f"""Answer the following based on the context.

Context:
{context}

Question: {question}
Answer:"""

    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return result.stdout.decode("utf-8").strip()
