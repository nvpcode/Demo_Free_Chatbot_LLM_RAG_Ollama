# 💬 NVP-Chatbot RAG Advanced | Trợ lý AI thông minh với Ollama + LangChain

Một hệ thống chatbot RAG tiên tiến tích hợp mô hình ngôn ngữ lớn Ollama với cơ chế Retrieval-Augmented Generation (RAG) nâng cao, bao gồm cache thông minh, reranking và đánh giá chất lượng.

## 🎯 Mục tiêu dự án

Xây dựng một chatbot thông minh có khả năng:
- Truy xuất và trả lời chính xác từ tài liệu người dùng cung cấp
- Sử dụng cache thông minh để tối ưu hiệu suất
- Hỗ trợ đa dạng định dạng tài liệu và phương pháp xử lý
- Đánh giá chất lượng chatbot RAG


## 🧠 Kiến trúc hệ thống
```

Người dùng ↔️ Streamlit UI ↔️ Cache Manager ↔️ LangChain Agent                       ↔️ Ollama LLM
                                              ↘️ VectorStore (FAISS + BM25) + Reranker

↘️ Evaluation Pipeline (RAGAS)
```

### **Core Components:**
- **LLM:** Ollama Llama3 (`incept5/llama3.1-claude:latest`)
- **Embedding Model:** OllamaEmbeddings (`nomic-embed-text:latest`)
- **Vector Store:** FAISS với hybrid retriever
- **Retriever:** Kết hợp FAISS (dense) + BM25 (sparse) với tỉ lệ 0.5:0.5
- **Reranker:** BAAI/bge-reranker-base để cải thiện chất lượng retrieval
- **Cache System:** Dual-thread caching với text similarity + vector similarity
- **Evaluation:** RAGAS metrics để đánh giá chất lượng

## 🚀 Tính năng chính

### **1. RAG Pipeline Nâng cao:**
- **Hybrid Retrieval:** Kết hợp dense (FAISS) và sparse (BM25) retrieval
- **Smart Reranking:** Sử dụng cross-encoder để sắp xếp lại kết quả
- **Flexible Chunking:** Hỗ trợ recursive và semantic chunking

### **2. Cache System Thông minh:**
- **Dual-Thread Caching:** 
  - Luồng 1: Text similarity (TF-IDF + fuzzy matching)
  - Luồng 2: Vector similarity (embedding-based)
- **Intelligent Cache Management:** Tự động dọn dẹp và tối ưu
- **Performance Optimization:** Giảm thời gian phản hồi và chi phí API

### **3. Đánh giá chất lượng:**
- **RAGAS Evaluation:** Context precision, recall, faithfulness, answer relevancy
- **Benchmark System:** Hệ thống đánh giá tự động cho RAG pipeline
- **Performance Metrics:** Theo dõi và cải thiện chất lượng câu trả lời

### **4. Giao diện người dùng:**
- **Streamlit Web UI:** Giao diện thân thiện và dễ sử dụng
- **Real-time Chat:** Hỗ trợ trò chuyện thời gian thực
- **File Upload:** Hỗ trợ .txt, .pdf, .docx, .xlsx
- **Chat History:** Lưu trữ và quản lý lịch sử hội thoại

## 🧱 Cấu trúc thư mục
```
├── main.py                         # Giao diện chính và điều khiển Streamlit
├── cache_communicate/             
│ └── caching_communicate.py        # Cache manager với dual-thread
├── models/                         
│ ├── llm.py                        # LLM interface và functions
│ └── embedder.py                   # Embedding model configuration
├── retriever/                      
│ └── retriever.py                  # Hybrid retriever (FAISS + BM25)
├── rerank/                         
│ └── reranker.py                   # Cross-encoder reranker
├── chunks_data/                    
│ ├── load_data.py                  # Data loading utilities
│ └── split_chunks.py               # Document chunking methods
├── benchmark_RAG/                  
│ ├── simulation_data_for_eval/     # Test data for evaluation
│ └── results_eval/                 # Evaluation results
├── ragas_eval.py                   # RAGAS evaluation pipeline
├── retrieval_eval.py               # Retrieval evaluation
├── generation_eval.py              # Generation evaluation
├── get_info_chat_history.py        # Chat history utilities
├── create_suggestions/             # Suggestion generation system
└── lib/                            # Library files and documentation
```

## ⚙️ Cài đặt và Chạy dự án

### **1. Yêu cầu hệ thống:**
- Python 3.10+
- Ollama server
- GPU (khuyến nghị cho embedding và reranking)

### **2. Cài đặt dependencies:**
```bash
pip install -r lib/requirements.txt
```

### **3. Cài đặt Ollama:**
```bash
# Tải và cài đặt Ollama từ https://ollama.com/
# Tải các models cần thiết:
ollama pull incept5/llama3.1-claude:latest
ollama pull nomic-embed-text:latest
```

### **4. Khởi động ứng dụng:**
```bash
streamlit run main.py
```

## 🔧 Cấu hình và Tùy chỉnh

### **Cache Configuration:**
- Điều chỉnh similarity threshold trong `cache_communicate.py`
- Tùy chỉnh cache size và expiry time
- Chọn phương pháp similarity (text, vector, hoặc cả hai)

### **Retrieval Tuning:**
- Thay đổi tỉ lệ weights giữa FAISS và BM25
- Điều chỉnh top_k cho retrieval và reranking
- Tùy chỉnh chunk size và overlap

### **Evaluation Setup:**
- Cấu hình RAGAS metrics
- Thiết lập ground truth data
- Chạy benchmark evaluation

## 📈 Đánh giá hiệu suất

### **RAGAS Metrics:**
- **Context Precision:** Độ chính xác của context được truy xuất
- **Context Recall:** Độ bao phủ của thông tin liên quan
- **Faithfulness:** Độ trung thực của câu trả lời với context
- **Answer Relevancy:** Độ liên quan của câu trả lời với câu hỏi

### **Performance Metrics:**
- Response time
- Cache hit rate
- Retrieval accuracy
- Memory usage


## Liên hệ
Có câu hỏi thắc mắc xin vui lòng liên hệ qua email: nguyenphuongv07@gmail.com.
