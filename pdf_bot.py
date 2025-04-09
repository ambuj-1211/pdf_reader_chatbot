import os

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from PyPDF2 import PdfReader

load_dotenv()
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_ENDPOINT"]=os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"]="Pdf_Chatbot"


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
        
        
def main():
    st.header("📄Chat with your pdf file")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunks = text_splitter.split_text(text=text)
        db = FAISS.from_texts(chunks, OllamaEmbeddings(model="gemma2:2b"))
        
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context.
        Think step by step before providing a detailed answer. Provide the complete answer which makes sense. Use points where necessary or use paragraph where necessary. Give clear and concise answers.
        <context>
        {context}
        </context>
        Question: {input}""")

        llm = OllamaLLM(model="gemma2:2b")
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever = VectorStoreRetriever(vectorstore=db)
        retrieval_chain=create_retrieval_chain(retriever,document_chain)
        
        query = st.text_input("Ask questions about your PDF file")
        if query:
            stream_handler = StreamHandler(st.empty())
            response = retrieval_chain.invoke({"input":query},callbacks=[stream_handler])
            st.write(response["answer"])
            

if __name__ == "__main__":
    main()