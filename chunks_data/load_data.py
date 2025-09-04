import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, UnstructuredFileLoader
import json
from langchain.schema import Document

# -----------------------------
# 1. Hàm load data
# -----------------------------
def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()  # lấy đuôi file
    
    if ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()

    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
        docs = loader.load()

    elif ext in [".xlsx", ".xls"]:
        loader = UnstructuredExcelLoader(file_path, mode="elements")
        docs = loader.load()

    else:
        # fallback cho các định dạng không rõ
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
    
    return docs
