from langchain.document_loader import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub



def run_rag(pdf_path, query):
    # Load the PDF document 
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Initialize embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6_v2")
    vector_store = Chroma.from_documents(pages, embeddings)

    retriver = vector_store.as_retriever()
    llm = HuggingFaceHub(repo_id ="google/flan-t5-base", model_kwargs={"temperature": 0.1, "max_length": 100})

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriver=retriver)
    return qa_chain.run(query)