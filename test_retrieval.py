from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_PATH = r"C:\Users\Pc\OneDrive\Desktop\nabeel python\Rag\project\vectorstore\engine_db"

def test_query(query):
    print("Loading embedding model...")
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Loading vector database...")
    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding
    )

    print(f"\nQuery: {query}\n")
    
    results = db.similarity_search(query, k=3)

    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print("-" * 50)
        print(doc.page_content[:800])  # preview first 800 chars


if __name__ == "__main__":
    user_query = input("Enter your question: ")
    test_query(user_query)