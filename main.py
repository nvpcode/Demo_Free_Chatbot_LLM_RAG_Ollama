import streamlit as st
import os
import json
import shutil
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from chunks_data.split_chunks import chunk_documents
from chunks_data.load_data import  load_data
from models.embedder import get_embedding_model
from retriever.retriever import get_hybrid_retriever
from rerank.reranker import get_rerank_from_retriever
from models.llm import get_answer_from_llm
from get_info_chat_history import get_info
# from create_suggestions.suggestions import generate_suggestions_from_docs

from cache_communicate.caching_communicate import CacheManager


# Thiết lập biến môi trường
model_LLM = "incept5/llama3.1-claude:latest"


# ----------------------------
# Setup Page
# ----------------------------
def setup_page():
    st.set_page_config(
        page_title="ChatBot_RAG",
        page_icon="💬",
        layout="wide"
    )


# ----------------------------
# Sidebar Config
# ----------------------------
def setup_sidebar():
    st.sidebar.header("1. Tải dữ liệu")
    uploaded_file = st.sidebar.file_uploader(
        "Chọn file dữ liệu (.txt, .pdf, .docx, .xlsx)",
        type=["txt", "pdf", "docx", "xlsx"],
        accept_multiple_files=True,
    )

    st.sidebar.header("2. Chia chunk")
    chunk_method = st.sidebar.selectbox(
        "Phương pháp chia chunk", ["recursive", "semantic"]
    )

    st.sidebar.header("3. AI Model")
    model_choice = st.sidebar.selectbox(
        "Chọn AI Model để trả lời:", [model_LLM]
    )

    return uploaded_file, chunk_method, model_choice


# ----------------------------
# Initialize app (load, chunk, embed, retriever, LLM)
# ----------------------------
def initialize_app(uploaded_file, chunk_method):
    if not uploaded_file:
        return None, None, None

    # Chunk
    chunk_size, overlap = 1024, 500
    all_chunks = []

    if uploaded_file:
        # Nếu thư mục tồn tại thì xóa
        if os.path.exists("tmp_data"):
            shutil.rmtree("tmp_data")
        os.makedirs("tmp_data", exist_ok=True)
        for uploaded_file in uploaded_file:
            file_path = os.path.join("tmp_data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"✅Đã lưu file vào {file_path}")

            # Load data
            docs = load_data(file_path)
            chunked_docs = chunk_documents(docs, method=chunk_method, chunk_size=chunk_size, overlap=overlap)
            all_chunks.extend(chunked_docs)

    st.info(f"📑 Đã chia thành {len(all_chunks)} chunk.")

    # Embedding
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(all_chunks, embeddings)

    # Retriever
    retriever_obj = get_hybrid_retriever(vector_store, chunked_docs, top_k=3, weights=(0.5, 0.5))

    # Rerank
    rerank_obj = get_rerank_from_retriever(retriever_obj, top_k=3)

    # Khởi tạo cache manager
    cache_manager = CacheManager()
    cache_manager.set_embedding_model(embeddings)

    return retriever_obj, rerank_obj, cache_manager


# ----------------------------
# Chat interface
# ----------------------------
def setup_chat_interface(model_choice):
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("💬 Chat-NVP")
    with col2:
        if st.button("🔄 Làm mới hội thoại"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"}
            ]
            st.session_state.input_enabled = True
            st.rerun()

    st.caption(f"🚀 Trợ lý AI được hỗ trợ bởi {model_choice}")

    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"}
        ]
        msgs.add_ai_message("Xin chào! Tôi có thể giúp gì cho bạn hôm nay?")

    # 👉 Hiển thị toàn bộ lịch sử hội thoại
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

    return msgs

# ----------------------------
# Handle user input
# ----------------------------
def handle_user_input(msgs, retriever_obj, rerank_obj, model_choice, cache_manager):
    if input_user := st.chat_input("Hãy nhập câu hỏi của bạn:"):
        # Lưu và hiển thị tin nhắn người dùng
        st.session_state.messages.append({"role": "human", "content": input_user})
        st.chat_message("human").write(input_user)
        msgs.add_user_message(input_user)

        # Lấy lịch sử chat gần nhất
        max_history_pairs = 5
        history_messages = st.session_state.messages[-(max_history_pairs * 2 + 1):-1]
        # Ghép thành chuỗi "Người: ...\nBot: ..."
        chat_history_str = "\n".join(
            f"{'Người dùng' if m['role'] == 'human' else 'Bot'}: {m['content']}"
            for m in history_messages
        )

        # Gọi AI xử lý và hiển thị

        # (Không Streaming)
        # with st.chat_message("assistant"):
        #     response = get_answer(chat_history_str, input_user, retriever, model_choice)

        #     # Hiển thị câu trả lời
        #     st.session_state.messages.append({"role": "assistant", "content": response})
        #     msgs.add_ai_message(response)
        #     st.write(response)

        # (Có Streaming)
        # Sử dụng cache
        with st.chat_message("assistant"):
            placeholder = st.empty()
            
            # Kiểm tra cache trước
            cached_answer = cache_manager.get_from_cache(input_user)
            
            if cached_answer:
                # Hiển thị câu trả lời từ cache
                placeholder.markdown(cached_answer)
                streamed_text = cached_answer
            else:
                # Gọi LLM nếu không có trong cache
                streamed_text = ""
                for token in get_answer_from_llm(chat_history_str, input_user, retriever_obj, rerank_obj, model_choice):
                    streamed_text += token
                    placeholder.markdown(streamed_text)
                
                # Lưu vào cache
                cache_manager.add_to_cache(input_user, streamed_text)
            
            # Lưu vào session
            st.session_state.messages.append({"role": "assistant", "content": streamed_text})
            msgs.add_ai_message(streamed_text)


# ----------------------------
# Main
# ----------------------------
def main():
    setup_page()
    uploaded_file, chunk_method, model_choice = setup_sidebar()

    # Khởi tạo app với cache
    retriever_obj, rerank_obj, cache_manager = initialize_app(uploaded_file, chunk_method)

    # ✅ Chỉ mở giao diện chat khi dữ liệu và agent đã sẵn sàng
    if retriever_obj is not None:
        msgs = setup_chat_interface(model_choice)
        handle_user_input(msgs, retriever_obj, rerank_obj, model_choice, cache_manager)

        # get_info("", retriever_obj, rerank_obj, model_choice)
    else:
        st.warning("⚠️ Vui lòng tải dữ liệu trước khi bắt đầu trò chuyện.")
    

if __name__ == "__main__":
    main()
