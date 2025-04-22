

# Rag Application 
A lightweight Retrieval-Augmented Generation (RAG) app that finds and generates meme captions based on **semantic meaning** — not just keywords.

Built with:
- **FAISS** (vector search)
- **HuggingFace Embeddings**
- **Ollama LLMs** (e.g.,DeepSeek)
- **Streamlit** (UI)


## ⚡ Quickstart

```bash
# 1. Clone
git clone https://github.com/your-username/semantic-meme-rag.git
cd semantic-meme-rag

# 2. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Prepare vectorstore
python src/data_pipeline.py

# 4. Launch app
streamlit run app.py
```
