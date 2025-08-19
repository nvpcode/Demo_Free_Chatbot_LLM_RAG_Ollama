from langchain_community.vectorstores import FAISS
import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_data_from_folder(folder_path="data"):
    """
    Quét toàn bộ thư mục và đọc tất cả các file file .txt, .docx, .pdf
    sau đó chia nhỏ nội dung thành các Document bằng RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    all_chunks = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)

    return all_chunks


def vector_store(all_chunks):
    import asyncio
    asyncio.set_event_loop(asyncio.new_event_loop())

    """
    Tạo và lưu vector embeddings vào FAISS từ file tài liệu.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://localhost:11434")
    
    vector_store = FAISS.from_documents(all_chunks, embeddings)

    return vector_store
