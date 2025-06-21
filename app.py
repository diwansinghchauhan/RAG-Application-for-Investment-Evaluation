import streamlit as st
import tempfile
from Pdfquery import extract_text_and_images
from Pdfquery import build_faiss_index, query_faiss
from Pdfquery import ask_mistral

st.set_page_config(page_title="Investor RAG Q&A", layout="centered")
st.title("RAG Application for Investment Evaluation")

uploaded_file = st.file_uploader("Upload pitch deck (PDF)", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    with st.spinner("🔍 Processing PDF..."):
        documents = extract_text_and_images(file_path)
        index, texts, model = build_faiss_index(documents)

    st.success("✅ Pitch deck uploaded and ready to use!")

    query = st.text_input("Ask a question about the pitch deck:")

    if st.button("Get Answer") and query:
        top_chunks = query_faiss(query, model, index, texts, k=3)
        context = "\n".join(top_chunks)
        with st.spinner("🧠 Thinking with Mistral..."):
            answer = ask_mistral(context, query)

        st.subheader("💬 Answer")
        st.write(answer)
