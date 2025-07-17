from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

def run_rag(pdf_path, query):
    loader = UnstructuredPDFLoader(pdf_path)
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever()
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.1, "max_length": 100})

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(query)
