# supabase_pdf_cohere.py
# Requires: pip install supabase pypdf langchain-cohere langchain-text-splitters \
#           langchain-core numpy requests python-dotenv
import os
import io
import logging
from typing import List

import numpy as np
import requests
from supabase import create_client
from pypdf import PdfReader
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── ENV Validation ───────────────────────────────────────────────────
def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Missing required environment variable: {name}. "
            f"Please set it in your .env file or environment."
        )
    return value

COHERE_API_KEY = _require_env("COHERE_API_KEY")
SUPABASE_URL = _require_env("SUPABASE_URL")
SUPABASE_KEY = _require_env("SUPABASE_KEY")

COHERE_EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-v4.0")
USE_FAISS = os.getenv("USE_FAISS", "false").lower() == "true"

# ─── PDF filenames ────────────────────────────────────────────────────
PDF_FILES = ["document.pdf"]

# ─── Batched Cohere Embeddings ────────────────────────────────────────
class BatchedCohereEmbeddings(CohereEmbeddings):
    """Wraps CohereEmbeddings to stay under the 96-text API batch limit."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if len(texts) <= 48:
            return super().embed_documents(texts)
        all_embeddings: List[List[float]] = []
        batch_size = 48
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(super().embed_documents(batch))
            logger.info(f"[EMBED] batch {i // batch_size + 1} done")
        return all_embeddings

embeddings = BatchedCohereEmbeddings(
    cohere_api_key=COHERE_API_KEY,
    model=COHERE_EMBED_MODEL,
)

# ─── Pure-numpy Cosine Retriever (no FAISS, no hangs) ────────────────
class CosineRetriever:
    """
    Lightweight in-memory retriever using cosine similarity.
    - Zero C++ dependencies
    - Works inside Uvicorn --reload / fastapi dev
    - Faster than FAISS for < 500 chunks
    """

    def __init__(
        self,
        texts: List[str],
        embedded_vectors: List[List[float]],
        metadatas: List[dict],
        embedding_model,
        k: int = 5,
    ):
        self.texts = texts
        self.metadatas = metadatas
        self.embedding_model = embedding_model
        self.k = k
        self.vectors = np.array(embedded_vectors, dtype=np.float32)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.normed = self.vectors / norms

    def invoke(self, query: str) -> List[Document]:
        q_vec = np.array(self.embedding_model.embed_query(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm
        scores = self.normed @ q_vec
        top_k_idx = np.argsort(scores)[-self.k :][::-1]
        docs: List[Document] = []
        for idx in top_k_idx:
            docs.append(
                Document(
                    page_content=self.texts[idx],
                    metadata=self.metadatas[idx],
                )
            )
        return docs

# ─── PDF Loader ───────────────────────────────────────────────────────
def load_pdfs_from_supabase() -> List[Document]:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    all_docs: List[Document] = []
    for filename in PDF_FILES:
        try:
            logger.info(f"[PDF] Fetching {filename}...")
            pdf_bytes = None
            try:
                pdf_bytes = supabase.storage.from_("pdfs").download(filename)
                logger.info("[PDF] Authenticated download OK")
            except Exception:
                url = supabase.storage.from_("pdfs").get_public_url(filename)
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                pdf_bytes = resp.content
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    all_docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": page_num,
                                "total_pages": len(reader.pages),
                            },
                        )
                    )
            logger.info(f"[PDF] Loaded {len(reader.pages)} pages from {filename} ✅")
        except Exception as e:
            logger.error(f"[PDF] Failed {filename}: {e}")
    return all_docs

# ─── Build Retriever ──────────────────────────────────────────────────
def build_retriever():
    logger.info("[RETRIEVER] Loading PDFs from Supabase...")
    docs = load_pdfs_from_supabase()
    if not docs:
        logger.warning("[RETRIEVER] No documents loaded.")
        return None

    chunk_size = 800 if COHERE_EMBED_MODEL.startswith("embed-v4") else 400
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"[RETRIEVER] {len(chunks)} chunks from {len(docs)} pages")
    if not chunks:
        return None

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    logger.info("[RETRIEVER] Embedding chunks with Cohere...")
    embedded_vectors = embeddings.embed_documents(texts)
    logger.info("[RETRIEVER] Embeddings received ✅")

    if USE_FAISS:
        logger.info("[RETRIEVER] Building FAISS index...")
        from langchain_community.vectorstores import FAISS
        vector_store = FAISS.from_documents(chunks, embeddings)
        ret = vector_store.as_retriever(search_kwargs={"k": 5})
        logger.info("[RETRIEVER] FAISS ready ✅")
        return ret

    logger.info("[RETRIEVER] Building CosineRetriever (hang-proof) ✅")
    return CosineRetriever(
        texts=texts,
        embedded_vectors=embedded_vectors,
        metadatas=metadatas,
        embedding_model=embeddings,
        k=5,
    )

if __name__ == "__main__":
    retriever = build_retriever()
    if retriever:
        results = retriever.invoke("example query")
        for doc in results:
            print(f"[{doc.metadata.get('source')} p{doc.metadata.get('page')}] {doc.page_content[:150]}...")