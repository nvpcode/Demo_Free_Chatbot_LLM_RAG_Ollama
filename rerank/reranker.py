from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


# Khởi tạo reranker với model BAAI/bge-reranker-large
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

def get_rerank_from_retriever(retriever_obj, top_k: int = 5):
    """
    Tạo rerank_obj từ retriever_obj sử dụng BAAI/bge-reranker-large.
    Chỉ lấy top_n kết quả sau khi rerank.
    """

    # Tạo CrossEncoderReranker
    compressor = CrossEncoderReranker(model=model, top_n=top_k)

    # Tạo ContextualCompressionRetriever
    rerank_obj = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=retriever_obj
    )
    return rerank_obj

def show_rerank_chunks(rerank_obj, query: str):
    """
    Hàm nhận retriever_obj và query đầu vào.
    Đầu ra: số lượng chunks và nội dung của từng chunk.
    """
    # Lấy danh sách các tài liệu liên quan
    reranked_docs = rerank_obj.invoke(query)

    print(f"Top docs after reranking:")

    # # In ra nội dung từng chunk
    # for i, doc in enumerate(reranked_docs, 1):
    #     print(f"--- Chunk {i} ---")
    #     print(doc.page_content.strip())
    #     if doc.metadata:
    #         print(f"📌 Metadata: {doc.metadata}")
    #     print()

    # In ra số lượng chunks
    print(f"🔎 Số lượng chunks lấy được: {len(reranked_docs)}")
    
    return reranked_docs