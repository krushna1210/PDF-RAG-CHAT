import os
import time
import cohere
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# ---------------------------
# Load environment
# ---------------------------
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "pdf-rag")

if not COHERE_API_KEY:
    raise ValueError("❌ Missing COHERE_API_KEY in environment")
if not PINECONE_API_KEY:
    raise ValueError("❌ Missing PINECONE_API_KEY in environment")

co = cohere.Client(COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)


# ---------------------------
# Embedding Helpers
# ---------------------------
def _embed_with_retry(
    texts,
    input_type="search_document",
    batch_size=4,          # ✅ small batch (safe for free tier)
    max_retries=6,
    sleep_base=3,
    progress_callback=None
):
    embeddings = []
    total = len(texts)
    done = 0
    skipped_batches = 0

    for i in range(0, total, batch_size):
        # ✅ trim long texts (Cohere has token limits)
        batch = [t[:900] for t in texts[i:i + batch_size]]

        for attempt in range(max_retries):
            try:
                resp = co.embed(
                    model=COHERE_EMBED_MODEL,
                    texts=batch,
                    input_type=input_type
                )
                embeddings.extend(resp.embeddings)
                done += len(batch)
                if progress_callback:
                    progress_callback(done, total)
                break  # success
            except cohere.errors.TooManyRequestsError:
                wait = sleep_base * (2 ** attempt)
                st.info(f"⚠️ Cohere rate limit. Retrying in {wait}s...")
                time.sleep(wait)
            except Exception as e:
                st.error(f"❌ Embedding error: {e}")
                break
        else:
            skipped_batches += 1
            st.warning(f"🚨 Skipped batch {i}-{i+batch_size} after retries")
            continue

        time.sleep(1.0)  # cooldown per batch

    if skipped_batches > 0:
        st.warning(f"⚠️ {skipped_batches} batches skipped. Some chunks not embedded.")

    return embeddings


def embed_texts_documents(texts, progress_callback=None):
    """For document chunks"""
    return _embed_with_retry(texts, input_type="search_document", progress_callback=progress_callback)


def embed_texts_queries(texts):
    """For user queries"""
    return _embed_with_retry(texts, input_type="search_query")


# ---------------------------
# Pinecone VectorStore
# ---------------------------
class PineconeVectorStore:
    def __init__(self):
        existing_indexes = [i["name"] for i in pc.list_indexes()]
        if PINECONE_INDEX not in existing_indexes:
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=1024,  # Cohere embed-multilingual-v3.0
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
            )
        self.index = pc.Index(PINECONE_INDEX)

    def add_embeddings(self, embeddings, metas, batch_size=100, progress_callback=None):
        """
        Add vectors to Pinecone with batching
        """
        total = len(embeddings)
        timestamp = int(time.time())

        for i in range(0, total, batch_size):
            batch_emb = embeddings[i:i + batch_size]
            batch_meta = metas[i:i + batch_size]

            vectors = []
            for j, (emb, meta) in enumerate(zip(batch_emb, batch_meta)):
                vectors.append({
                    "id": f"id-{timestamp}-{i+j}",
                    "values": emb,
                    "metadata": meta
                })

            self.index.upsert(vectors=vectors)

            if progress_callback:
                progress_callback(i + len(batch_emb), total)

            time.sleep(0.5)  # ✅ small delay to avoid Pinecone request size issues

    def search(self, query_emb, top_k=5):
        results = self.index.query(
            vector=query_emb,
            top_k=top_k,
            include_metadata=True
        )
        return [(m.metadata, m.score) for m in results.matches]
