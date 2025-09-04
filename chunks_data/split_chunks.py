from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from models.embedder import get_embedding_model
from typing import List
from langchain.schema import Document

# -----------------------------
# Hàm chia chunk
# -----------------------------
def chunk_documents(documents: List[Document], method: str = "recursive", 
                    chunk_size: int = 1024, overlap: int = 200) -> List[Document]:
    """
    Hàm chia nhỏ tài liệu thành nhiều đoạn (chunk).
    
    Tham số:
        - documents: danh sách các Document đầu vào
        - method: phương pháp chia ("recursive" hoặc "semantic")
        - chunk_size: độ dài mỗi chunk (số ký tự, dùng cho recursive)
        - overlap: số ký tự chồng lấn giữa các chunk (dùng cho recursive)
    
    Hai phương pháp chính:
    1. Recursive (RecursiveCharacterTextSplitter):
        - Cắt văn bản theo độ dài ký tự (chunk_size).
        - Có hỗ trợ chồng lấn (overlap) để tránh mất ngữ cảnh khi câu bị cắt ngang.
        - Ưu điểm: nhanh, đơn giản, không cần model.
        - Nhược điểm: có thể cắt ngang câu, mất nghĩa.

    2. Semantic (SemanticChunker):
        - Dùng mô hình embedding để xác định điểm ngắt tự nhiên trong văn bản (dựa trên ngữ nghĩa).
        - Cắt tại những chỗ nội dung thay đổi nhiều → giữ nguyên ý nghĩa từng chunk.
        - Ưu điểm: chunk có ý nghĩa tốt hơn, phù hợp cho RAG.
        - Nhược điểm: chậm hơn, cần gọi model embedding.
    
    Trả về:
        - Danh sách Document đã được chia nhỏ
    """
    if method.endswith("semantic"):
        # Lấy embedding model để phục vụ chia semantic
        embeddings = get_embedding_model()
        splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

        chunked_docs = []
        for doc in documents:
            # Chia văn bản thành nhiều chunk theo ngữ nghĩa
            chunks = splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                chunked_docs.append(
                    Document(page_content=chunk, metadata={**doc.metadata, "chunk_id": i})
                )
        return chunked_docs

    else:  # recursive/sliding window
        # Chia văn bản dựa trên độ dài ký tự
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        return splitter.split_documents(documents)