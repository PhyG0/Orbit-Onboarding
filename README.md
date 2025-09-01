# Onboarding Buddy (RAG-powered FAQ Assistant)

A lightweight Flask service that answers new-hire questions using your official onboarding documents (PDF/MD/TXT) with source citations. It builds embeddings from files in docs/, retrieves the most relevant chunks, and asks an LLM to answer only from that context.

# ✨ Features

RAG pipeline over docs/ (PDF, Markdown, Text)
Cited answers (filenames included)
Admin endpoints for health and reindex
Zero frontend required (ships with a simple index.html)
Render-friendly deployment (or run locally)

# 🧠 How it works (high level)

Read all files from docs/ → convert to plain text (pypdf + Markdown-to-text).
Chunk text and generate embeddings using text-embedding-3-small.
For each question, retrieve top-k chunks (cosine similarity).
Send a constrained system prompt + retrieved context to the chat model (gpt-4o-mini by default).
Return a concise answer + source filenames.


# 🧰 Tech Stack

Flask, Gunicorn
OpenAI Python SDK (chat + embeddings)
NumPy, PyPDF, Markdown

# 📁 Content Guidelines (docs/)

Supported: .pdf, .md, .txt
Keep documents official and current (handbooks, policies, IT guides).
After adding/replacing files, call POST /api/reindex or restart.



