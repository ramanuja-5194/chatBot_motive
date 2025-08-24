import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.parser import parse_document
from langchain_huggingface import HuggingFaceEmbeddings

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
        # Apply text splitting here
        # splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=500,      # size of each chunk
        #     chunk_overlap=100,   # overlap so context isn’t lost
        #     length_function=len,
        #     separators=["\n\n", "\n", ". ", " "]  # split smartly at sentence/paragraph boundaries
        # )
        # chunks = splitter.split_text(text)

        # for chunk in chunks:
        #     documents.append(Document(page_content=chunk, metadata=metadata))

    if not documents:
        raise ValueError("No valid documents found to embed. Check parsing logic.")

    # Use a strong embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-roberta-large-v1"
        # Alternative: "intfloat/e5-large-v2" if you want a very strong retriever
        # sentence-transformers/all-mpnet-base-v2
        # sentence-transformers/all-roberta-large-v1
    )

    # Vectorstore
    db = FAISS.from_documents(documents, embeddings)
    db.save_local("vectorstore")
    print(f"✅ Ingestion completed, stored {len(documents)} chunks.")

if __name__ == "__main__":
    ingest_docs()
