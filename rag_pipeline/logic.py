import pdfplumber
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.docstore.document import Document

def run_rag(pdf_path, query):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([full_text])

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.1, "max_length": 100})

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain.run(query)
