import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# =========================
# Streamlit Config
# =========================
st.set_page_config(
    page_title="PDF RAG vs RAG + HyDE",
    layout="wide"
)
st.title("📄 Ask Your PDF — RAG vs RAG + HyDE")

# =========================
# API KEY
# =========================
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("Please set GROQ_API_KEY in Streamlit Secrets")
    st.stop()

# =========================
# Upload PDF
# =========================
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        st.success(f"Loaded {len(docs)} pages")

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        splits = splitter.split_documents(docs)

        # Embeddings + Vector Store
        embeddings = FastEmbedEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )
        retriever = vectorstore.as_retriever()

    # =========================
    # LLMs
    # =========================
    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=api_key,
        temperature=0.5,
        max_tokens=512
    )

    hyde_llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=api_key,
        temperature=0
    )

    # =========================
    # Prompts
    # =========================
    rag_prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question.
If you don't know the answer, say "I don't know".

Context:
{context}

Question:
{question}
""")

    hyde_prompt = ChatPromptTemplate.from_template("""
You are an expert assistant.
Generate a detailed hypothetical answer that could appear in a document.

Question:
{question}

Hypothetical Answer:
""")

    # =========================
    # HyDE Chain
    # =========================
    hyde_chain = (
        hyde_prompt
        | hyde_llm
        | StrOutputParser()
    )

    def hyde_retriever(question):
        hypothetical_answer = hyde_chain.invoke({"question": question})
        return retriever.invoke(hypothetical_answer)

    # =========================
    # Chains
    # =========================
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    rag_hyde_chain = (
        {
            "context": RunnableLambda(hyde_retriever),
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # =========================
    # UI
    # =========================
    question = st.text_input("Ask a question about the PDF")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Get Answer (RAG)"):
            if question:
                with st.spinner("Generating answer..."):
                    answer = rag_chain.invoke(question)
                    st.subheader("📘 RAG Answer")
                    st.write(answer)

    with col2:
        if st.button("Get Answer (RAG + HyDE)"):
            if question:
                with st.spinner("Generating answer with HyDE..."):
                    answer = rag_hyde_chain.invoke(question)
                    st.subheader("🚀 RAG + HyDE Answer")
                    st.write(answer)

else:
    st.info("Upload a PDF to start")
