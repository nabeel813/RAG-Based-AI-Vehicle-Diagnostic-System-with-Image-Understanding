# RAG-Based-AI-Vehicle-Diagnostic-System-with-Image-Understanding
🚗 Multimodal Retrieval-Augmented Vehicle Diagnostic Assistant

An AI-powered vehicle diagnostic system that combines image understanding, semantic search, conversational memory, and large language models to provide structured and intelligent vehicle fault analysis.

📌 Overview

The Multimodal Retrieval-Augmented Vehicle Diagnostic Assistant is a final-year AI project that integrates:

🧠 Large Language Models (LLaMA 3.3 via Groq)

🔎 Retrieval-Augmented Generation (RAG)

🖼 Image Captioning (BLIP)

🗂 Vector Database (Chroma)

💬 Conversational Memory

📄 PDF Report Generation

The system accepts both text queries and vehicle images, retrieves relevant manual data, and generates a structured technical diagnosis.

🎯 Problem Statement

Traditional vehicle diagnostics depend heavily on human expertise and manual inspection. There is no unified AI assistant that:

Understands vehicle images

Retrieves relevant manual information

Maintains conversation context

Generates structured diagnostic reports

This project solves that gap using a multimodal RAG-based architecture.

🏗 System Architecture
User Input (Text / Image)
        ↓
Streamlit Web Interface
        ↓
BLIP (Image Captioning)
        ↓
Vector Database (Chroma)
        ↓
LLaMA 3.3 (Groq API)
        ↓
Structured Diagnostic Output
        ↓
PDF Report Export
🧠 Core Technologies
Component	Technology Used
Frontend	Streamlit
LLM	LLaMA 3.3 70B (Groq API)
Image Captioning	BLIP
Embeddings	MiniLM-L6-v2
Vector Database	Chroma
PDF Generation	ReportLab
🔎 How RAG Works in This Project

User submits a query.

The system generates embeddings.

Chroma retrieves relevant manual documents.

Retrieved context is combined with conversation history.

LLaMA generates a structured response.

This improves:

Accuracy

Reliability

Technical correctness

Reduction of hallucinations

💬 Conversational Memory

The assistant stores recent conversation history to:

Support follow-up questions

Maintain issue context

Improve multi-step diagnostics

Example:

User: “The piston is broken.”
User: “Can I continue driving?”

The assistant remembers the previous issue and responds accordingly.

📄 Output Format

The system provides structured responses including:

Identified Issue

Possible Causes

Recommended Fix

Safety Advice

It also allows exporting the diagnosis as a PDF report.

🚀 Features

Multimodal Input (Text + Image)

Semantic Search using Vector Database

Context-aware Conversation

Structured Technical Diagnosis

PDF Report Export

Clean Professional UI
