import streamlit as st
from src.processing import create_rag_chain

@st.cache_resource
def get_chain():
    return create_rag_chain()

st.set_page_config(page_title="Meme RAG Bot", layout="centered")
st.title("🤖 Meme Finder")
query = st.text_input("Ask about memes...", help="Ex: Show memes about exams")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Finding relevant memes..."):
            try:
                response = get_chain()({"query": query})
                st.subheader("Response")
                st.write(response["result"])
                
                sources = {doc.metadata["url"] for doc in response["source_documents"]}
                if sources:
                    st.subheader("📌 Related Memes")
                    for url in sources:
                        st.markdown(f"- [{url}]({url})")
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("*Powered by DeepSeek + FAISS + Streamlit*")