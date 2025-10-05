# PDF-RAG Chat Application

An intelligent chatbot application that leverages **Retrieval-Augmented Generation (RAG)** to interact with uploaded PDF documents. The app enables users to upload one or more PDFs, extract and embed their content, and ask questions in natural language to receive contextually accurate answers.

---

## 🚀 Features

- 📄 **Multi-PDF Upload** – Upload one or more PDF files at once.
- 🧠 **RAG-based Answering** – Combines document retrieval with large language model (LLM) reasoning.
- ⚡ **Fast Response Time** – Optimized with vector embeddings and local caching.
- 🔍 **Semantic Search** – Retrieves the most relevant chunks of text from documents.
- 💬 **Interactive Chat UI** – User-friendly Streamlit interface for smooth conversation.
- 🔐 **Local Execution** – Runs entirely on your system (no external API required).

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend | Streamlit |
| Backend | FastAPI / Python |
| Embeddings | Cohere (embed-multilingual-v3.0) |
| Vector Store | Pinecone |
| LLM | Groq or Local LLaMA-based model |
| File Handling | PyMuPDF, LangChain Document Loaders |

---

## 📁 Project Structure

```bash
pdf-rag-chat/
├── backend/
│   ├── app.py               # FastAPI backend logic
│   ├── config.py            # Configuration file (API keys, settings)
│   ├── generator.py         # Embedding + LLM generator
│   ├── requirements.txt     # Backend dependencies
│   └── dataset/             # Embedded vector storage
│
├── frontend/
│   ├── app.py               # Streamlit app entry point
│   ├── pages/               # Additional Streamlit pages
│   └── assets/              # UI assets, icons, etc.
│
├── README.md                # Project documentation
└── .gitignore               # Ignore unnecessary files
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/pdf-rag-chat.git
cd pdf-rag-chat
```

### 2️⃣ Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 4️⃣ Configure Environment Variables
Create a `.env` file in the backend directory with the following keys (examples provided — **do not commit real keys**):

```text
# Cohere (Embeddings)
COHERE_API_KEY=your_cohere_api_key
COHERE_EMBED_MODEL=embed-multilingual-v3.0

# Pinecone (Vector store)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=us-east-1
PINECONE_INDEX=pdf-rag

# Groq / LLM provider (if used)
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile

# Optional / runtime
STREAMLIT_SERVER_PORT=8501
```

> Tip: Add `/backend/.env` (or `.env`) to `.gitignore` and provide a `backend/.env.example` with placeholder values for contributors.

### 5️⃣ Run the Application
#### Run Backend (FastAPI)
```bash
cd backend
uvicorn app:app --reload
```

#### Run Frontend (Streamlit)
```bash
cd frontend
streamlit run app.py
```

---

## 🧠 How It Works

1. **PDF Upload** → The user uploads PDF(s).
2. **Text Extraction** → PyMuPDF extracts clean text.
3. **Chunking** → Text is split into manageable chunks.
4. **Embedding Generation** → Cohere generates dense vectors for chunks.
5. **Vector Indexing** → Pinecone indexes embeddings for semantic retrieval.
6. **RAG Query** → The chatbot retrieves the most relevant chunks and uses the LLM to craft contextual answers.

---

## 🖼️ Architecture Overview

graph TD
A[User Uploads PDFs] --> B[Text Extraction]
B --> C[Chunking]
C --> D[Embedding Generation]
D --> E[Vector Database (Pinecone)]
E --> F[Query + Retrieval]
F --> G[LLM Response Generation]
G --> H[Answer Display on UI]
```

---

## 💡 Future Enhancements

- 📚 Add support for **DOCX and TXT** file formats.
- 🌐 Integrate **multilingual question answering**.
- 🧩 Add **chat memory and conversation context**.
- 🔊 Enable **voice input/output**.
- ☁️ Optional **cloud deployment** with Docker & CI/CD pipeline.

---

## 🪪 License

This project is licensed under the **MIT License** – feel free to use, modify, and distribute.
