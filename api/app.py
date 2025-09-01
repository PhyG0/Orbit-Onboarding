import os, re, glob
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from pypdf import PdfReader
import markdown as md
from openai import OpenAI

# ============================
# (Dev only) load .env.local if present
# ============================
try:
    from dotenv import load_dotenv
    load_dotenv(".env.local")
except Exception:
    pass

# ======================================================
# 🔑 OpenAI Configuration (from environment)
# ======================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Set it in your host's Environment Variables.")

# Choose defaults (can be overridden in /api/ask by `mode`)
DEFAULT_MODE = "fast"   # fast | quality | cheap
CHAT_MODELS = {
    "fast":    "gpt-4o-mini",
    "quality": "gpt-4.1-mini",
    "cheap":   "gpt-4o-mini"
}
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536  # embedding size for text-embedding-3-small

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ======================================================
# 📂 Project paths (keep folders next to this file)
# ======================================================
APP_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(APP_DIR, "static")
DOCS_DIR   = os.path.join(APP_DIR, "docs")

# Flask app
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")

# ======================================================
# 🧠 Global memory (vectors)
# ======================================================
VECTORS = None        # np.ndarray [n, d]
CHUNKS = []           # list[str]
SOURCES = []          # list[str]
INDEX_READY = False   # lazy-build flag

# ======================================================
# 🔧 Helper functions
# ======================================================
def read_txt_md(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    text = md.markdown(text)
    text = re.sub("<[^<]+?>", " ", text)
    return text

def read_pdf(path):
    reader = PdfReader(path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n".join(pages)

def load_documents():
    docs = []
    if not os.path.isdir(DOCS_DIR):
        return docs
    for path in glob.glob(os.path.join(DOCS_DIR, "**/*"), recursive=True):
        if os.path.isdir(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in [".md", ".txt"]:
            text = read_txt_md(path)
        elif ext == ".pdf":
            text = read_pdf(path)
        else:
            continue
        if text.strip():
            docs.append({"path": path, "text": text})
    return docs

def chunk_text(text, chunk_size=900, overlap=150):
    words = text.split()
    out, i = [], 0
    while i < len(words):
        j = min(i + chunk_size, len(words))
        out.append(" ".join(words[i:j]))
        if j == len(words):
            break
        i = max(0, j - overlap)
    return out

def embed_texts(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.vstack([np.array(e.embedding, dtype=np.float32) for e in resp.data])

def cosine_sim(all_vecs, qvec):
    all_norm = all_vecs / (np.linalg.norm(all_vecs, axis=1, keepdims=True) + 1e-10)
    qnorm = qvec / (np.linalg.norm(qvec) + 1e-10)
    return np.dot(all_norm, qnorm)

def build_index():
    """(Re)build the in-memory index. Safe to call multiple times."""
    global VECTORS, CHUNKS, SOURCES, INDEX_READY
    CHUNKS, SOURCES = [], []
    docs = load_documents()
    for d in docs:
        for ch in chunk_text(d["text"]):
            CHUNKS.append(ch)
            SOURCES.append(os.path.basename(d["path"]))
    if CHUNKS:
        VECTORS = embed_texts(CHUNKS)
    else:
        VECTORS = np.zeros((0, EMBED_DIM), dtype=np.float32)  # keep embed dim stable
    INDEX_READY = True

def ensure_index():
    """Lazy build to avoid heavy work during cold boot."""
    global INDEX_READY
    if not INDEX_READY:
        build_index()

def search_top_k(query, k=5):
    ensure_index()
    if VECTORS is None or VECTORS.shape[0] == 0:
        return []
    qvec = embed_texts([query])[0]
    sims = cosine_sim(VECTORS, qvec)
    idx = np.argsort(-sims)[:k]
    return [{"text": CHUNKS[i], "source": SOURCES[i], "score": float(sims[i])} for i in idx]

def resolve_chat_model(mode: str) -> str:
    return CHAT_MODELS.get((mode or "").lower(), CHAT_MODELS.get(DEFAULT_MODE))

def call_llm(system_prompt, user_prompt, mode=None):
    model = resolve_chat_model(mode)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt}
            ],
            temperature=0.2
        )
        return resp.choices[0].message.content, model
    except Exception as e:
        # Return an error message to the client (helps with logs on Render)
        return f"LLM error: {e}", model

# ======================================================
# 🚀 Cold start indexing (best-effort, non-fatal)
# ======================================================
try:
    # Optionally skip initial build; lazy build will handle first request
    pass
except Exception as e:
    print("Index build error at startup (will lazy-build later):", e)

# ======================================================
# 🌐 Routes
# ======================================================
@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    mode  = (data.get("mode")  or "").strip()
    if not query:
        return jsonify({"answer": "Please type a question."})
    results = search_top_k(query, k=5)
    context = "\n\n---\n\n".join([f"[Source: {r['source']}]\n{r['text']}" for r in results])

    system_prompt = (
        "You are the Onboarding Buddy for new employees. "
        "Answer using ONLY the provided company context. "
        "If the answer is not in the context, say you’re not sure and suggest who to contact (HR or IT). "
        "Be concise and actionable."
    )
    user_prompt = f"User question:\n{query}\n\nContext:\n{context}\n\nCite filenames if relevant."
    answer, resolved_model = call_llm(system_prompt, user_prompt, mode=mode or DEFAULT_MODE)
    srcs = sorted(set([r["source"] for r in results]))
    return jsonify({"answer": answer, "sources": srcs, "model": resolved_model})

@app.route("/api/reindex", methods=["POST"])
def reindex():
    build_index()
    return jsonify({"status": "ok", "message": "Reindexed in memory.", "chunks": len(CHUNKS)})

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "embedding_model": EMBED_MODEL,
        "chat_models": CHAT_MODELS,
        "default_mode": DEFAULT_MODE,
        "index_ready": INDEX_READY,
        "chunks": len(CHUNKS)
    })

# Serve UI pages
@app.route("/")
def root():
    return send_from_directory(APP_DIR, "index.html")

@app.route("/admin")
def admin_page():
    return send_from_directory(APP_DIR, "admin.html")

# ======================================================
# 🏁 Run locally
# ======================================================
if __name__ == "__main__":
    # For local dev; on Render you'll run via:
    # gunicorn app:app --bind 0.0.0.0:$PORT
    app.run(host="127.0.0.1", port=8000, debug=True)
