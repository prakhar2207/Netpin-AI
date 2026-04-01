# 🤖 Netpin AI Assistant (RAG-based DevOps Chatbot)

An intelligent DevOps support assistant built using Retrieval-Augmented Generation (RAG).  
It helps users query Netpin documentation, debug issues, and get step-by-step solutions through a conversational interface.

---

## 🚀 Features

- 🔍 Hybrid Retrieval (FAISS + BM25)
- 🧠 Context-aware responses (RAG pipeline)
- 💬 Conversational memory (multi-turn chat)
- ⚡ LLM via Hugging Face API (Llama 3)
- 🎯 Re-ranking with CrossEncoder
- 🌐 Streamlit Chat UI (ChatGPT-like)
- 📚 Source attribution for answers
- 🛠️ DevOps-focused assistance

---

## 🏗️ Architecture

User Query  
↓  
Query Rewriting (Memory-aware)  
↓  
Hybrid Retrieval (FAISS + BM25)  
↓  
Re-ranking (CrossEncoder)  
↓  
Context Building  
↓  
LLM (Llama 3 via Hugging Face API)  
↓  
Response + Sources  

---

## 📂 Project Structure

Netpin-Bot/
│
├── app.py              # Streamlit UI  
├── rag.py              # RAG pipeline (retrieval + LLM)  
├── ingest.py           # Data ingestion + FAISS indexing  
├── db/                 # Vector database (generated)  
├── .env                # API keys  
├── requirements.txt    # Dependencies  
└── README.md  

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

git clone https://github.com/prakhar2207/Netpin-AI.git  
cd Netpin-AI  

---

### 2️⃣ Create Virtual Environment

conda create -n rag_env python=3.10  
conda activate rag_env  

---

### 3️⃣ Install Dependencies

pip install -r requirements.txt  

---

### 4️⃣ Add Environment Variables

Create a `.env` file:

HF_TOKEN=your_huggingface_api_token  

---

### 5️⃣ Run Data Ingestion

python ingest.py  

This will:
- Scrape documentation  
- Clean and chunk text  
- Generate embeddings  
- Store data in FAISS  

---

### 6️⃣ Run the App

streamlit run app.py  

---

## 🧠 How It Works

### 🔹 Retrieval
- FAISS → semantic search  
- BM25 → keyword search  

### 🔹 Re-ranking
- CrossEncoder improves relevance  

### 🔹 Generation
- Llama 3 via Hugging Face API  

### 🔹 Memory
- Maintains chat history  
- Supports follow-up queries  

---

## 💡 Example Queries

- How to setup Netpin?  
- Why did my deployment fail?  
- How to fix CrashLoopBackOff?  
- Explain Infrastructure Debt Index  

---

## ⚠️ Notes

- Uses faiss-cpu (stable on Windows)  
- LLM runs via API (no local GPU required)  
- GPU optional for embeddings and reranker  

---

## 🚀 Future Improvements

- FastAPI backend  
- Web integration  
- Streaming responses  
- Multi-user sessions  
- Analytics dashboard  

---

## 🛠️ Tech Stack

- Python 3.10  
- LangChain  
- FAISS  
- Hugging Face API  
- Sentence Transformers  
- Streamlit  

---

## 🤝 Contributing

Pull requests are welcome.  
For major changes, please open an issue first.

---

## 📜 License

MIT License  

---

## 👨‍💻 Author

Prakhar Shukla  

---

## ⭐ If you like this project

Give it a star on GitHub ⭐