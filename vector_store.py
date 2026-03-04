import json
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHUNKS_PATH = r"C:\Users\Pc\OneDrive\Desktop\nabeel python\Rag\project\data\processed\chunks.json"
DB_PATH = r"C:\Users\Pc\OneDrive\Desktop\nabeel python\Rag\project\vectorstore\engine_db"


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        documents.append(
            Document(
                page_content=item["content"],
                metadata=item["metadata"]
            )
        )

    return documents


def create_vector_db():
    print("Loading chunks...")
    documents = load_chunks()

    print("Creating embeddings model...")
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating vector database...")
    db = Chroma.from_documents(
        documents,
        embedding,
        persist_directory=DB_PATH
    )

    db.persist()
    print("Vector database created successfully!")


if __name__ == "__main__":
    os.makedirs("vectorstore", exist_ok=True)
    create_vector_db()

