import streamlit as st
import os
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from seed_data import load_data_from_folder, vector_store 
from llm import get_retriever, get_answer

# Thiết lập biến môi trường
model_LLM = "incept5/llama3.1-claude:latest"

def setup_page():
    st.set_page_config(
        page_title="ChatBot_RAG",
        page_icon="💬",
        layout="wide"
    )

def initialize_app():
    setup_page()

def setup_sidebar():
    with st.sidebar:
        st.title("⚙️Cấu hình")
        
        st.header("📑Embeddings Model")
        embeddings_choice = st.radio(
            "Chọn Embeddings Model:",
            ["nomic-embed-text:latest"]
        )

        st.header("🐧Model LLM")
        model_choice = st.radio(
            "Chọn AI Model để trả lời:",
            [model_LLM]
        )

        st.header("📁Tải dữ liệu")
        uploaded_files = st.file_uploader(
            "Chọn một hoặc nhiều file dữ liệu .txt, docx, pdf",
            type=["txt", "docx", "pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            os.makedirs("data", exist_ok=True)
            for uploaded_file in uploaded_files:
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                st.success(f"✅Đã lưu file vào {file_path}")

        return embeddings_choice, model_choice

def setup_chat_interface(model_choice):
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("💬 Chat-NVP")
    with col2:
        if st.button("🔄 Làm mới hội thoại"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"}
            ]
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

def handle_user_input(msgs, retriever, model_choice):
    if prompt := st.chat_input("Hãy nhập câu hỏi của bạn:"):
        # Lưu và hiển thị tin nhắn người dùng
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

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
        #     response = get_answer(chat_history_str, prompt, retriever, model_choice)

        #     # Hiển thị câu trả lời
        #     st.session_state.messages.append({"role": "assistant", "content": response})
        #     msgs.add_ai_message(response)
        #     st.write(response)

        # (Có Streaming)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            streamed_text = ""
            for token in get_answer(chat_history_str, prompt, retriever, model_choice):
                streamed_text += token
                placeholder.markdown(streamed_text)
            
            # Lưu vào session
            st.session_state.messages.append({"role": "assistant", "content": streamed_text})
            msgs.add_ai_message(streamed_text)      


def main():
    initialize_app()
    _, model_choice = setup_sidebar()

    # Giao diện chat
    msgs = setup_chat_interface(model_choice)

    # Vector hóa dữ liệu
    all_chunks = load_data_from_folder()

    if not all_chunks:
        st.warning("📂Chưa có dữ liệu để tạo embeddings. Vui lòng upload file .txt vào sidebar.")
        return  # Dừng lại, không chạy tiếp main
    else:
        vector = vector_store(all_chunks)
        retriever = get_retriever(vector)

    # Chat
    handle_user_input(msgs, retriever, model_choice)

if __name__ == "__main__":
    main()