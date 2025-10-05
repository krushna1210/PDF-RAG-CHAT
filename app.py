import streamlit as st
import time
import os
from dotenv import load_dotenv

from helpers import extract_text_from_pdf, chunk_text
from vectorstore import PineconeVectorStore, embed_texts_documents, embed_texts_queries
from groq import Groq

# ---------------------------
# Load environment
# ---------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# Streamlit Config
# ---------------------------
st.set_page_config(page_title="📚 DocuChat", layout="centered", initial_sidebar_state="collapsed")

# ---------------------------
# UI Styling
# ---------------------------
st.markdown(
    """
    <style>
    .chat-history {
        max-height: 65vh;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid var(--border-color, #ddd);
        border-radius: 12px;
        background-color: var(--background-color, #fafafa);
        margin-bottom: 80px;
    }
    .user-msg {
        background-color: #DCF8C6;
        padding: 10px 14px;
        border-radius: 12px;
        margin: 6px 0;
        text-align: right;
        float: right;
        clear: both;
        max-width: 80%;
        word-wrap: break-word;
    }
    .assistant-msg {
        background-color: #F1F0F0;
        padding: 10px 14px;
        border-radius: 12px;
        margin: 6px 0;
        text-align: left;
        float: left;
        clear: both;
        max-width: 80%;
        word-wrap: break-word;
    }
    .file-pill {
        display: inline-block;
        padding: 6px 12px;
        margin: 6px 0;
        border-radius: 12px;
        background-color: #e5f1fb;
        font-size: 14px;
        color: #0366d6;
        border: 1px solid #b3d4fc;
    }
    .chat-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 12px;
        background: var(--background-color, white);
        border-top: 1px solid var(--border-color, #ddd);
    }
    [data-theme="light"] {
        --background-color: #fafafa;
        --border-color: #ddd;
    }
    [data-theme="dark"] {
        --background-color: #1e1e1e;
        --border-color: #444;
    }

    /* ✅ Fix alignment of input + button */
    div[data-baseweb="input"] > div {
        height: 42px !important;
    }
    div[data-baseweb="input"] input {
        height: 42px !important;
        padding: 0 12px !important;
    }
    .stButton > button {
        width: 120px !important;
        height: 42px !important;
        font-size: 16px !important;
        border-radius: 10px !important;
        margin-top: 0px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Title
# ---------------------------
st.markdown("<h1>📚 DocuChat</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px;'>Your AI-powered assistant for chatting with PDFs</p>", unsafe_allow_html=True)

# ---------------------------
# Session State
# ---------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None


# ---------------------------
# Process PDF
# ---------------------------
def process_pdf(file):
    tmp_path = f"uploaded_{int(time.time())}.pdf"
    with open(tmp_path, "wb") as f:
        f.write(file.read())

    pages = extract_text_from_pdf(tmp_path)
    chunks = chunk_text(pages, max_chars=1200, overlap=200)

    texts = [c["text"] for c in chunks]
    metas = [{"chunk_id": c["chunk_id"], "text": c["text"], "page": c["page"]} for c in chunks]

    # ✅ Progress bar for embedding
    embed_progress = st.progress(0, text="🔎 Generating embeddings...")
    def embed_callback(done, total):
        embed_progress.progress(done / total, text=f"🔎 Embedding chunks... ({done}/{total})")

    embeddings = embed_texts_documents(texts, progress_callback=embed_callback)

    # ✅ Progress bar for Pinecone upload
    pine_progress = st.progress(0, text="📡 Uploading to Pinecone...")
    def pine_callback(done, total):
        pine_progress.progress(done / total, text=f"📡 Uploading vectors... ({done}/{total})")

    vs = PineconeVectorStore()
    vs.add_embeddings(embeddings, metas, batch_size=100, progress_callback=pine_callback)

    embed_progress.empty()
    pine_progress.empty()

    return vs


# ---------------------------
# Answer Question
# ---------------------------
def answer_question(question: str, top_k: int = 5):
    vs = st.session_state.vectorstore

    context = ""
    if vs:
        q_vec = embed_texts_queries([question])[0]
        results = vs.search(q_vec, top_k=top_k)

        context_snippets = []
        for meta, dist in results:
            snippet = meta["text"]
            page = meta.get("page", None)
            context_snippets.append(f"[Page {page}]\n{snippet}\n")
        context = "\n\n---\n\n".join(context_snippets)

    if context.strip():
        prompt = f"""
You are **DocuChat**, a helpful assistant. 
Answer the user’s question using the document context below. 
- Provide structured, clear answers  
- Use bullet points or short paragraphs  
- Be concise and conversational  
- If answer not in context, say: *"I could not find that information in the document."*  

📄 Document Context:
{context}

❓ User Question: {question}
"""
    else:
        prompt = f"""
You are **DocuChat**, a friendly AI assistant. 
Even without a document, you can chat naturally.  
Keep answers concise, structured, and helpful.  

❓ User Question: {question}
"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are DocuChat, a smart and friendly assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error calling Groq API: {e}"


# ---------------------------
# UI Flow
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])

if uploaded_file and st.button("Process PDF"):
    st.session_state.doc_name = uploaded_file.name
    with st.spinner("Processing PDF..."):
        st.session_state.vectorstore = process_pdf(uploaded_file)
    st.success(f"✅ {uploaded_file.name} processed successfully!")

if st.session_state.doc_name:
    st.markdown(f"<div class='file-pill'>📄 {st.session_state.doc_name}</div>", unsafe_allow_html=True)

st.markdown("---")

# Chat history
st.markdown("<div class='chat-history'>", unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>{msg['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-msg'>{msg['text']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Input + Send button
st.markdown("<div class='chat-input'>", unsafe_allow_html=True)
col1, col2 = st.columns([10, 2])
with col1:
    user_q = st.text_input("Type your message here...", key="input_q", placeholder="Ask anything...")
with col2:
    send_clicked = st.button("Send")
st.markdown("</div>", unsafe_allow_html=True)

if send_clicked and user_q:
    st.session_state.chat_history.append({"role": "user", "text": user_q})
    with st.spinner("Thinking..."):
        ans = answer_question(user_q, top_k=5)
    st.session_state.chat_history.append({"role": "assistant", "text": ans})
    st.rerun()

if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
