"""
knowledge/rag_store.py

Builds a local, free vector store over the maintenance procedure docs
in /procedures using sentence-transformers embeddings + FAISS.
No hosted embedding API required.

Requires: pip install sentence-transformers faiss-cpu langchain-community
"""
import os
import glob
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

PROCEDURES_DIR = os.path.join(os.path.dirname(__file__), "..", "procedures")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")

# Small, fast, fully local embedding model — no API key needed.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_procedure_docs() -> List[Document]:
    docs = []
    for path in sorted(glob.glob(os.path.join(PROCEDURES_DIR, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append(Document(page_content=text, metadata={"source": os.path.basename(path)}))
    if not docs:
        raise FileNotFoundError(f"No procedure docs found in {PROCEDURES_DIR}")
    return docs


def build_index(force_rebuild: bool = False) -> FAISS:
    """Build (or load a cached) FAISS index over the procedure docs."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if not force_rebuild and os.path.exists(INDEX_DIR):
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    raw_docs = _load_procedure_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)

    index = FAISS.from_documents(chunks, embeddings)
    os.makedirs(INDEX_DIR, exist_ok=True)
    index.save_local(INDEX_DIR)
    return index


_index_instance = None

def get_index() -> FAISS:
    global _index_instance
    if _index_instance is None:
        _index_instance = build_index()
    return _index_instance


def retrieve_procedures(query: str, k: int = 3) -> List[dict]:
    """Semantic search over maintenance procedures. Returns the top-k
    chunks with their source filename, for the RAG tool to hand to the LLM."""
    index = get_index()
    results = index.similarity_search_with_score(query, k=k)
    return [
        {"source": doc.metadata["source"], "content": doc.page_content, "score": float(score)}
        for doc, score in results
    ]


if __name__ == "__main__":
    # Quick manual test — builds the index on first run
    build_index(force_rebuild=True)
    hits = retrieve_procedures("pressure sensor PS1 reading is unstable")
    for h in hits:
        print(f"[{h['source']}] score={h['score']:.3f}\n{h['content'][:200]}...\n")
