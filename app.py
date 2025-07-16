import streamlit as st
from rag_pipeline.logic import run_rag
import os

st.title("📘 Ask Questions on Your PDF")

uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")
query = st.text_input("Enter your question")

if uploaded_pdf and query:
    temp_path = "temp_uploaded.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_pdf.read())

    answer = run_rag(temp_path, query)
    st.write("💡 Answer:", answer)


