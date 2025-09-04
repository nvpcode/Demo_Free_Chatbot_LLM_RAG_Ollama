import json
import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from sklearn.metrics import ndcg_score

from retriever.retriever import get_hybrid_retriever
from rerank.reranker import get_rerank_from_retriever
from models.embedder import get_embedding_model


# ====================
# METRICS
# ====================
def precision_at_k(retrieved_ids, relevant_ids, k):
    """Precision@K: tỷ lệ tài liệu đúng trong top-K."""
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
    return relevant_retrieved / k

def recall_at_k(retrieved_ids, relevant_ids, k):
    """Recall@K: bao nhiêu tài liệu đúng được lấy ra trên tổng số đúng."""
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
    return relevant_retrieved / len(relevant_ids) if relevant_ids else 0.0

def hit_at_k(retrieved_ids, relevant_ids, k):
    """Hit@K: có ít nhất một tài liệu đúng trong top-K."""
    retrieved_k = retrieved_ids[:k]
    return 1.0 if any(doc_id in relevant_ids for doc_id in retrieved_k) else 0.0

def mrr(retrieved_ids, relevant_ids):
    """MRR: reciprocal rank của tài liệu đúng đầu tiên."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def map_at_k(retrieved_ids, relevant_ids, k):
    """MAP: trung bình precision theo thứ tự khi gặp tài liệu đúng."""
    score = 0.0
    hits = 0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            hits += 1
            score += hits / i
    return score / len(relevant_ids) if relevant_ids else 0.0

def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """NDCG@K: đánh giá thứ hạng tài liệu."""
    y_true = [[1 if doc_id in relevant_ids else 0 for doc_id in retrieved_ids[:k]]]
    y_score = [[len(retrieved_ids) - i for i in range(len(retrieved_ids[:k]))]]
    return ndcg_score(y_true, y_score)

def context_recall(retrieved_ids, relevant_ids, k):
    """Context Recall: giả định đạt nếu có ít nhất một tài liệu đúng."""
    return hit_at_k(retrieved_ids, relevant_ids, k)

# ====================
# HÀM ĐÁNH GIÁ
# ====================
def evaluate_retriever(retriever, ground_truth):
    results = []
    for q, relevant_ids in ground_truth.items():
        retrieved_docs = retriever.invoke(q)
        retrieved_ids = [doc.metadata["id"] for doc in retrieved_docs]
        k=len(retrieved_ids)

        metrics = {
            "query": q,
            "precision@k": precision_at_k(retrieved_ids, relevant_ids, k),
            "recall@k": recall_at_k(retrieved_ids, relevant_ids, k),
            "hit@k": hit_at_k(retrieved_ids, relevant_ids, k),
            "mrr": mrr(retrieved_ids, relevant_ids),
            "map": map_at_k(retrieved_ids, relevant_ids, k),
            "ndcg@k": ndcg_at_k(retrieved_ids, relevant_ids, k),
            "context_recall": context_recall(retrieved_ids, relevant_ids, k),
            "retrieved": retrieved_ids,
        }
        results.append(metrics)
    return results

# ====================
# HÀM MAIN
# ====================
if __name__ == "__main__":

    # ĐỌC DỮ LIỆU
    # ====================
    with open("simulation_data_for_eval/documents.json", "r", encoding="utf-8") as f:
        docs_data = json.load(f)
    documents = [Document(page_content=d["content"], metadata={"id": d["id"]}) for d in docs_data]

    with open("simulation_data_for_eval/ground_truth.json", "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # EMBEDDING DỮ LIỆU VÀ LƯU VÀO VECTOR STORE
    # ====================
    embedding_model = get_embedding_model()
    vector_store = FAISS.from_documents(documents, embedding_model)

    # TẠO HYBRID RETRIEVER --> RERANK
    # ====================
    hybrid_retriever_obj = get_hybrid_retriever(vector_store, documents, top_k=5, weights=(0.5, 0.5))
    rerank_obj = get_rerank_from_retriever(hybrid_retriever_obj, top_k=3)

    # CHẠY HÀM ĐÁNH GIÁ
    # ====================
    results = evaluate_retriever(rerank_obj, ground_truth)

    # Lưu kết quả ra CSV
    # ====================
    df = pd.DataFrame(results)
    df.to_csv("results_eval/Result_retriever_evaluation.csv", index=False, encoding="utf-8-sig")

