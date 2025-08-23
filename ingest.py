import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from utils.parser import parse_document
from langchain.embeddings import HuggingFaceEmbeddings
load_dotenv()

def ingest_docs():
    docs = parse_document("data/document.txt")
    documents = []

    for d in docs:
        text = d["content"].strip()
        if not text:   # skip empty chunks
            continue

        metadata = {
            "title": d["title"],
            "url": d["url"],
            "categories": d["categories"],
            "tags": d["tags"],
            "content_type": d["content_type"],
        }
        documents.append(Document(page_content=text, metadata=metadata))

    if not documents:
        raise ValueError("No valid documents found to embed. Check parsing logic.")

    # 🔹 HuggingFace embeddings (runs locally, no token limit issues)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Vectorstore
    db = FAISS.from_documents(documents, embeddings)
    db.save_local("vectorstore")
    print(f"✅ Ingestion completed, stored {len(documents)} documents.")

if __name__ == "__main__":
    ingest_docs()
