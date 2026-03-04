import os
import streamlit as st
from groq import Groq
from PIL import Image
from datetime import datetime

# LangChain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Vehicle Diagnostic Assistant",
    page_icon="🚗",
    layout="wide"
)

# =====================================================
# CLEAN PROFESSIONAL HEADER (FIXED VERSION)
# =====================================================
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
    <style>
    .animated-header {
        font-size: 38px;
        font-weight: 700;
        background: linear-gradient(90deg, #1F77B4, #00B4D8, #1F77B4);
        background-size: 200% auto;
        color: transparent;
        background-clip: text;
        -webkit-background-clip: text;
        animation: shine 4s linear infinite;
    }

    .sub-text {
        color: #888;
        font-size: 15px;
        margin-top: -8px;
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }
    </style>

    <div class="animated-header">
        🚗 AI Vehicle Diagnostic Assistant
    </div>
    <div class="sub-text">
        Multimodal | RAG-Powered | Conversational AI
    </div>
    """, unsafe_allow_html=True)

with col2:
    current_time = datetime.now().strftime("%d %b %Y | %H:%M:%S")

    st.markdown(f"""
    <div style="
        background-color:#E6F4EA;
        color:#137333;
        padding:8px 12px;
        border-radius:20px;
        font-weight:600;
        font-size:14px;
        text-align:center;">
        🟢 System Online
    </div>
    <div style="text-align:center; font-size:13px; color:#888; margin-top:6px;">
        {current_time}
    </div>
    """, unsafe_allow_html=True)

st.divider()


# =====================================================
# PDF GENERATOR
# =====================================================
def generate_pdf_report(user_query, image_caption, diagnosis):
    file_path = "vehicle_diagnostic_report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Vehicle Diagnostic Report", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("<b>User Query:</b>", styles["Heading3"]))
    elements.append(Paragraph(user_query, styles["BodyText"]))
    elements.append(Spacer(1, 0.2 * inch))

    if image_caption:
        elements.append(Paragraph("<b>Image Description:</b>", styles["Heading3"]))
        elements.append(Paragraph(image_caption, styles["BodyText"]))
        elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("<b>AI Diagnosis:</b>", styles["Heading3"]))
    elements.append(
        Paragraph(diagnosis.replace("\n", "<br/>"), styles["BodyText"])
    )

    doc.build(elements)
    return file_path


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("## ⚙ Controls")

    uploaded_image = st.file_uploader(
        "Upload Vehicle Image",
        type=["png", "jpg", "jpeg"]
    )

    if st.button("🧹 Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Model: LLaMA 3.3 70B")
    st.caption("Embeddings: MiniLM-L6-v2")
    st.caption("Vector DB: Chroma")


# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_vector_db():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory="vectorstore/engine_db",
        embedding_function=embedding
    )
    return db


@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model


db = load_vector_db()
blip_processor, blip_model = load_blip()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# =====================================================
# MEMORY SYSTEM
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =====================================================
# USER INPUT
# =====================================================
user_text = st.chat_input("Ask about vehicle issues...")


# =====================================================
# PROCESS REQUEST
# =====================================================
if user_text or uploaded_image:

    query_text = ""
    image_description = ""

    # IMAGE ANALYSIS
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            inputs = blip_processor(image, return_tensors="pt")
            output = blip_model.generate(**inputs)
            image_description = blip_processor.decode(
                output[0], skip_special_tokens=True
            )

        st.info(f"🧠 Image Description: {image_description}")
        query_text += image_description + ". "

    # TEXT INPUT
    if user_text:
        query_text += user_text
        st.session_state.messages.append(
            {"role": "user", "content": user_text}
        )

    # CONVERSATION MEMORY
    recent_messages = st.session_state.messages[-6:]
    conversation_context = "\n".join(
        [f"{msg['role'].upper()}: {msg['content']}" for msg in recent_messages]
    )

    final_query = conversation_context + "\nCURRENT QUESTION:\n" + query_text

    # RAG RETRIEVAL
    docs = db.similarity_search(final_query, k=5)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant manual data found."

    prompt = f"""
You are a professional vehicle diagnostic assistant.

Conversation:
{conversation_context}

Manual Context:
{context}

Question:
{query_text}

Provide:
1. Identified Issue
2. Possible Causes
3. Recommended Fix
4. Safety Advice
"""

    with st.spinner("Generating diagnostic report..."):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

    assistant_reply = response.choices[0].message.content

    st.success("✅ Diagnostic Analysis Complete")

    pdf_path = generate_pdf_report(
        user_query=query_text,
        image_caption=image_description,
        diagnosis=assistant_reply
    )

    with open(pdf_path, "rb") as file:
        st.download_button(
            label="📄 Download Diagnostic Report (PDF)",
            data=file,
            file_name="vehicle_diagnostic_report.pdf",
            mime="application/pdf"
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )

    with st.chat_message("assistant"):
        st.markdown(assistant_reply)