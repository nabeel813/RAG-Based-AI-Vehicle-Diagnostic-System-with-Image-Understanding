import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW_PATH = r"C:\Users\Pc\OneDrive\Desktop\nabeel python\Rag\project\data\Raw\merged.pdf"
PROCESSED_PATH = r"C:\Users\Pc\OneDrive\Desktop\nabeel python\Rag\project\data\raw\processed"


def load_and_chunk_pdf():
    print("Loading PDF...")

    loader = PyPDFLoader(RAW_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    return chunks


PROCESSED_PATH = r"C:\Users\Pc\OneDrive\Desktop\nabeel python\Rag\project\data\processed\chunks.json"

def save_chunks(chunks):
    os.makedirs(r"C:\Users\Pc\OneDrive\Desktop\nabeel python\Rag\project\data\processed", exist_ok=True)

    cleaned_chunks = []
    for chunk in chunks:
        cleaned_chunks.append({
            "content": chunk.page_content,
            "metadata": chunk.metadata
        })

    with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned_chunks, f, indent=4)

    print("Chunks saved successfully.")


if __name__ == "__main__":
    chunks = load_and_chunk_pdf()
    save_chunks(chunks)