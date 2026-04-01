import faiss
import torch
import os
from dotenv import load_dotenv
from functools import lru_cache
from collections import deque

# LangChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Retrieval
from rank_bm25 import BM25Okapi

# Re-ranking
from sentence_transformers import CrossEncoder

# ✅ Hugging Face API
from huggingface_hub import InferenceClient

# =========================
# ENV
# =========================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# =========================
# EMBEDDINGS
# =========================
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"}
    )

# =========================
# LOAD FAISS
# =========================
def load_faiss():
    embeddings = get_embeddings()

    vs = FAISS.load_local(
        "db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    try:
        vs.index = faiss.index_cpu_to_all_gpus(vs.index)
        print("✅ FAISS running on GPU")
    except:
        print("⚠️ FAISS CPU mode")

    return vs

# =========================
# BM25
# =========================
def build_bm25(docs):
    tokenized = [doc.page_content.split() for doc in docs]
    return BM25Okapi(tokenized), docs

# =========================
# ✅ LOAD LLAMA 3 API
# =========================
def load_llm():
    print("🌐 Using Meta Llama 3 via Hugging Face API")

    return InferenceClient(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        token=HF_TOKEN
    )

# =========================
# PREPROCESS
# =========================
def preprocess(query):
    return query.lower().strip()

# =========================
# FORMAT CONTEXT
# =========================
def format_context(docs):
    context = ""
    for i, doc in enumerate(docs):
        context += f"[Source {i+1}]\n{doc.page_content}\n\n"
    return context

# =========================
# Rewriting Querry
# =========================
def rewrite_query(client, query, history):
    history_text = "\n".join(history[-6:])

    prompt = f"""
You are an AI assistant that rewrites user queries into standalone questions.

Conversation:
{history_text}

Follow-up question:
{query}

Rewrite it into a complete standalone question:
"""

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1
        )

        return response.choices[0].message["content"].strip()

    except:
        return query  # fallback

# =========================
# PROMPT
# =========================
def build_prompt(context, query, history):
    history_text = "\n".join(history[-6:])
    return f"""
You are Netpin AI — a friendly and professional DevOps virtual assistant.

Conversation so far:
{history_text}

Your role:
- Help users understand Netpin features, errors, and setup
- Guide them step-by-step like a real support engineer
- Be conversational, helpful, and clear

Behavior guidelines:
- Greet naturally when appropriate (e.g., first interaction)
- Be polite and supportive
- Explain things in simple terms
- Provide actionable steps when solving problems
- Format answers clearly (use bullet points or steps if needed)
- Keep responses concise but complete

STRICT RULES:
- Answer ONLY using the provided context
- Do NOT make up information
- If the answer is not found, say: "I couldn't find this in the documentation. Can you provide more details?"

Response style:
- Start with a helpful sentence (not robotic)
- Then give explanation
- Then actionable steps (if applicable)

Context:
{context}

User Question:
{query}

Assistant Answer:
"""

# =========================
# ✅ GENERATE (LLAMA 3 CHAT)
# =========================
def generate(client, prompt):
    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a DevOps assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,            
            temperature=0.2,
            top_p=0.9,
            stream=False
        )

        return response.choices[0].message["content"].strip()

    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# RAG SYSTEM
# =========================
class RAGSystem:
    def __init__(self):
        print("🔄 Loading FAISS...")
        self.vs = load_faiss()

        print("🔄 Preparing BM25...")
        self.all_docs = list(self.vs.docstore._dict.values())
        self.bm25, self.bm25_docs = build_bm25(self.all_docs)

        print("🔄 Loading re-ranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        print("🔄 Loading LLM (Llama 3 API)...")
        self.client = load_llm()
        self.chat_history = deque(maxlen=8)

    # ---------------------
    # HYBRID RETRIEVAL
    # ---------------------
    def retrieve(self, query, k=5):
        query = preprocess(query)

        faiss_docs = self.vs.similarity_search(query, k=k)

        scores = self.bm25.get_scores(query.split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        bm25_docs = [self.bm25_docs[i] for i in top_idx]

        return faiss_docs + bm25_docs

    # ---------------------
    # RERANK
    # ---------------------
    def rerank(self, query, docs, top_k=5):
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    # ---------------------
    # MAIN ASK
    # ---------------------
    @lru_cache(maxsize=100)
    def ask(self, query):
        # 🔥 Step 1: Rewrite query using history
        rewritten_query = rewrite_query(self.client, query, list(self.chat_history))

        print(f"\n🔍 Rewritten Query: {rewritten_query}")

        # 🔥 Step 2: Retrieval
        docs = self.retrieve(rewritten_query)
        docs = self.rerank(rewritten_query, docs)

        context = format_context(docs)
        
        if len(context.strip()) < 50:
            answer = "I couldn't find this in documentation. Can you provide more details?"
        else:
            # 🔥 Step 3: Build prompt with memory
            prompt = build_prompt(context, query, list(self.chat_history))

            # 🔥 Step 4: Generate answer
            answer = generate(self.client, prompt)
        
        if query.lower() in ["hi", "hello", "hey"]:
            return {
                "answer": "Hey! 👋 I'm your Netpin AI assistant. How can I help you today?",
                "sources": [],
                "confidence": 1
            }

        # 🔥 Step 5: Save memory
        self.chat_history.append(f"User: {query}")
        self.chat_history.append(f"Assistant: {answer}")

        sources = [d.metadata.get("source", "") for d in docs]

        return {
            "answer": answer,
            "sources": sources,
            "confidence": 0.9
        }

# =========================
# RUN
# =========================
if __name__ == "__main__":
    rag = RAGSystem()

    while True:
        q = input("\n💬 Ask: ")
        if q.lower() in ["exit", "quit"]:
            break

        res = rag.ask(q)

        print("\n🤖 Answer:\n", res["answer"])
        print("\n📚 Sources:")
        for s in res["sources"]:
            print("-", s)