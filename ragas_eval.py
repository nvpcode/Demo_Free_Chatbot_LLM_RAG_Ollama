import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from datasets import Dataset
import json

# Load dữ liệu
with open("benchmark_RAG/simulation_data_for_eval/documents.json", "r", encoding="utf-8") as f:
    documents = {doc["id"]: doc["content"] for doc in json.load(f)}

with open("benchmark_RAG/simulation_data_for_eval/ground_truth.json", "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

with open("benchmark_RAG/simulation_data_for_eval/chat_history.json", "r", encoding="utf-8") as f:
    chat_history = {item["question"]: {"answer": item["answer"], "context": item["context"]} for item in json.load(f)}

# Tạo dataset theo định dạng RAGAS
data_samples = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": []
}

for item in ground_truth:
    q = item["question"]
    gt_ids = item["ground_truth"]

    # Thêm dữ liệu vào data_samples
    data_samples["question"].append(q)

    # Lấy câu trả lời từ chat_history
    data_samples["answer"].append(chat_history.get(q, {}).get("answer", ""))

    # Lấy context từ chat_history (list các đoạn text)
    data_samples["contexts"].append(chat_history.get(q, {}).get("context", []))

    # ground_truth: nối nội dung thật từ documents
    data_samples["ground_truth"].append(" ".join([documents[doc_id] for doc_id in gt_ids]))

# Lưu ra file JSON
output_file = "benchmark_RAG/simulation_data_for_eval/data_samples.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data_samples, f, ensure_ascii=False, indent=2)

print(f"✅ Đã lưu data_samples vào {output_file}")


dataset = Dataset.from_dict(data_samples)


# Thực thi đánh giá
result = evaluate(
    dataset=dataset,
    metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
)

# In kết quả
print(result)