from langchain.schema import Document
import fitz 
from PIL import Image
import io
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load TrOCR (OCR model from Hugging Face)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")  # or handwritten
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")

# Optional: Use GPU if available
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    documents = []

    for page_index in range(len(doc)):
        page = doc[page_index]

        # Text from PDF page
        page_text = page.get_text().strip()
        if page_text:
            documents.append(Document(
                page_content=page_text,
                metadata={"source": f"page_{page_index + 1}"}
            ))

        # Images from page
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # OCR using TrOCR
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            ocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if ocr_text.strip():
                documents.append(Document(
                    page_content=ocr_text.strip(),
                    metadata={"source": f"image_page_{page_index+1}_img_{img_index+1}"}
                ))

    return documents


# Build FAISS vector index from documents
def build_faiss_index(documents):
    texts = [doc.page_content for doc in documents]
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(384)
    index.add(np.array(embeddings).astype("float32"))
    return index, texts, model


# Perform similarity search
def query_faiss(query, model, index, texts, k=3):
    q_vec = model.encode([query]).astype("float32")
    D, I = index.search(q_vec, k)
    return [texts[i] for i in I[0]]


# Query Mistral via Ollama
import subprocess

def ask_mistral(context, question, model_name="mistral"):
    prompt = f"""Answer the following based on the context.

Context:
{context}

Question: {question}
Answer:"""

    result = subprocess.run(
        ["ollama", "run", model_name],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return result.stdout.decode("utf-8").strip()
