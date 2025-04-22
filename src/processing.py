import json
import os
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama


def process_data():
    data_path = "data/semantic_memes.json"
    index_path = "faiss_index"
    
    if not os.path.exists(data_path):
        os.makedirs("data", exist_ok=True)
        print("Downloading dataset...")
        ds = load_dataset("bhavyagiri/semantic-memes")["train"]
        ds = ds.select(range(min(len(ds), 2000)))
        
        records = [{"input": item["input"], "url": item["url"]} for item in ds]
        with open(data_path, "w") as f:
            json.dump(records, f)
        print(f"Saved dataset to {data_path}")

    # Create vectorstore if missing
    if not os.path.exists(index_path):
        print("Creating vectorstore...")
        with open(data_path) as f:
            records = json.load(f)  
        
        docs = [Document(
            page_content=r["input"], 
            metadata={"url": r["url"]}
        ) for r in records]

        FAISS.from_documents(
            docs,
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        ).save_local(index_path)


def create_rag_chain():
    #Create and return the RAG chain
    process_data()  
    
    return RetrievalQA.from_chain_type(
        llm=Ollama(model="deepseek-r1:1.5b"),
        retriever=FAISS.load_local(
            "faiss_index",
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        ).as_retriever(),
        chain_type="stuff",
        return_source_documents=True
    )

if __name__ == "__main__":
    # Run data processing when executed directly
    process_data()
    print("Data processing complete. Run app.py next.")