"""
StudyMate - Streamlit PDF Q&A App

Dependencies:
  pip install streamlit PyMuPDF sentence-transformers faiss-cpu requests numpy scikit-learn
  (On Windows you might prefer: conda install -c conda-forge faiss-cpu)

Replace the LLM call in `call_llm_with_context` with your Watsonx / other model API details.
"""

import streamlit as st
import fitz  # PyMuPDF
import tempfile
import os
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import json
from sklearn.neighbors import NearestNeighbors

# Try to import faiss, otherwise plan to use sklearn fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# ---------------------------
# Utility: text extraction
# ---------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    for page in doc:
        texts.append(page.get_text())
    doc.close()
    return "\n".join(texts)

# ---------------------------
# Utility: chunking (simple)
# ---------------------------
def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    """
    Splits text into chunks roughly max_chars long, breaking on newlines/sentence boundaries.
    """
    import re
    sentences = re.split(r'(?<=[\.\?\!]\s)', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current += s
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s
    if current.strip():
        chunks.append(current.strip())
    return chunks

# ---------------------------
# Embedding and Index
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def build_index(embeddings: np.ndarray):
    if FAISS_AVAILABLE:
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings.astype(np.float32))
        return ("faiss", index)
    else:
        nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(embeddings)
        return ("sklearn", nn)

def query_index(index_tuple, embeddings: np.ndarray, query_emb: np.ndarray, top_k: int = 5):
    kind, index = index_tuple
    if kind == "faiss":
        D, I = index.search(query_emb.astype(np.float32), top_k)
        return I[0], D[0]
    else:
        # sklearn fallback
        distances, indices = index.kneighbors(query_emb, n_neighbors=top_k)
        return indices[0], distances[0]

# ---------------------------
# LLM call (placeholder)
# ---------------------------
def call_llm_with_context(question: str, context_chunks: List[str]) -> str:
    """
    Use IBM's Granite API to generate answers based on the provided context.
    """
    # Get API key and endpoint from environment variable or Streamlit secrets
    API_KEY = os.getenv("IBM_GRANITE_API_KEY") or st.secrets.get("IBM_GRANITE_API_KEY")
    API_ENDPOINT = os.getenv("IBM_GRANITE_API_ENDPOINT") or st.secrets.get("IBM_GRANITE_API_ENDPOINT", 
                           "https://bam-api.res.ibm.com/v1/generate")
    
    if not API_KEY:
        return ("IBM Granite API key not configured. Please set IBM_GRANITE_API_KEY "
                "in .streamlit/secrets.toml file.")
    
    # Compose the prompt
    prompt = (
        "You are StudyMate, an academic assistant. Use the provided context to answer questions accurately. "
        "If the answer cannot be found in the context, say you don't know.\n\n"
        "Context:\n" + "\n---\n".join(context_chunks[:6]) + "\n\n"
        "Question: " + question + "\n\n"
        "Answer: "
    )
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model_id": "granite-13b-chat-v2",  # or your preferred IBM model
        "inputs": [prompt],
        "parameters": {
            "temperature": 0.1,
            "max_new_tokens": 500,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
            "stop_sequences": []
        }
    }
    
    try:
        # Call the IBM Granite API
        response = requests.post(
            API_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        if "results" in result and len(result["results"]) > 0:
            return result["results"][0]["generated_text"].strip()
        else:
            return "Error: Unexpected response format from API"
            
    except Exception as e:
        return f"LLM request failed: {str(e)}"

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="StudyMate", layout="wide")
st.title("StudyMate — PDF Conversational Q&A")

with st.sidebar:
    st.markdown("## Upload PDFs")
    uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
    st.markdown("---")
    st.markdown("## Settings")
    embed_model_name = st.text_input("Embedding model (sentence-transformers)", value="all-MiniLM-L6-v2")
    top_k = st.slider("Number of retrieved chunks (k)", 1, 10, 5)
    st.markdown("**Note:** Set your IBM Granite API key in the `.streamlit/secrets.toml` file as `IBM_GRANITE_API_KEY`.")

col1, col2 = st.columns([2, 3])

with col1:
    st.header("Documents")
    if uploaded_files:
        all_text = []
        for f in uploaded_files:
            bytes_data = f.read()
            st.write(f"Uploaded: {f.name} — {len(bytes_data)} bytes")
            text = extract_text_from_pdf_bytes(bytes_data)
            st.markdown(f"**{f.name}** — Extracted {len(text)} characters.")
            all_text.append(text)
        corpus_text = "\n\n".join(all_text)
        st.success("Extraction complete.")
    else:
        st.info("No PDFs uploaded yet.")
        corpus_text = ""

    if st.button("Preview extracted text (first 2000 chars)"):
        st.text(corpus_text[:2000])

with col2:
    st.header("Ask a question")
    question = st.text_area("Enter your question", height=120)
    if st.button("Get Answer"):
        if not uploaded_files:
            st.error("Please upload at least one PDF.")
        elif not question.strip():
            st.error("Please type a question.")
        else:
            with st.spinner("Preparing embeddings and index..."):
                # chunk corpus
                chunks = chunk_text(corpus_text, max_chars=800)
                st.write(f"Created {len(chunks)} chunks from documents.")
                # load model
                embed_model = load_embedding_model(embed_model_name)
                # embed chunks in batches
                batch_size = 64
                embeddings = []
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    emb = embed_model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
                    embeddings.append(emb)
                embeddings = np.vstack(embeddings)
                # build index
                index_tuple = build_index(embeddings)

                # embed query
                q_emb = embed_model.encode([question], convert_to_numpy=True)

                # query
                idxs, dists = query_index(index_tuple, embeddings, q_emb, top_k=top_k)

                retrieved = [chunks[i] for i in idxs]
                st.subheader("Retrieved context (top results)")
                for i, (r, dist) in enumerate(zip(retrieved, dists)):
                    st.markdown(f"**Result {i+1} (score: {float(dist):.4f})**")
                    st.write(r[:800] + ("..." if len(r) > 800 else ""))

            with st.spinner("Calling LLM to generate an answer..."):
                answer = call_llm_with_context(question, retrieved)
                st.subheader("Answer")
                st.write(answer)

st.markdown("---")
st.markdown("### Troubleshooting / Notes")
st.markdown("""
- If you see `ModuleNotFoundError`: install packages listed at top.
- For Windows, installing `faiss-cpu` via `pip` sometimes fails — use conda: `conda install -c conda-forge faiss-cpu`.
- Replace the LLM call with your provider's exact API schema and endpoint.
""")
