import os
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ==============================
# 🔐 SET YOUR GROQ API KEY
# ==============================
os.environ["GROQ_API_KEY"] = ""

# ==============================
# 🔹 LOAD EMBEDDING MODEL
# ==============================
print("Loading embedding model...")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==============================
# 🔹 LOAD VECTOR DATABASE
# ==============================
print("Loading vector database...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, r"C:\Users\Pc\OneDrive\Desktop\nabeel python\Rag\project\vectorstore\engine_db", r"C:\Users\Pc\OneDrive\Desktop\nabeel python\Rag\project\vectorstore\engine_db")

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embedding
)

# ==============================
# 🔹 RETRIEVE CONTEXT
# ==============================
def retrieve_context(db, query, k=5):
    results = db.similarity_search(query, k=k)

    if not results:
        return None

    context_text = ""
    for doc in results:
        context_text += "\n\n" + doc.page_content

    return context_text


# ==============================
# 🔹 GENERATE ANSWER
# ==============================
def generate_answer(context, query):
    client = Groq()

    prompt = f"""
You are a technical assistant.
Answer ONLY using the given context.
If answer is not in context, say:
"Information not available in database."

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


# ==============================
# 🔹 MAIN LOOP
# ==============================
if __name__ == "__main__":

    print("\nAsk your engine question (type 'exit' to quit):")

    while True:
        query = input("\n> ")

        if query.lower() == "exit":
            break

        context = retrieve_context(db, query)

        if context is None:
            print("\nNo relevant information found in the database.")
            continue

        answer = generate_answer(context, query)

        print("\n" + "=" * 70)
        print("AI ANSWER:\n")
        print(answer)
        print("=" * 70)