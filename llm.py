from langchain_ollama import ChatOllama
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate

def get_retriever(vector_store):
    """
    Kết hợp FAISS retriever (dựa trên embedding) và BM25 retriever (dựa trên từ khóa)
    theo tỷ trọng 7:3.
    """
    ## FAISS-based retriever (embedding)
    retriever_faiss = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    ## BM25-based retriever (keyword)

        #Lấy 100 documents gần nhất từ FAISS
    top_docs = vector_store.similarity_search("", k=100)

        # Tạo BM25 retriever từ 100 tài liệu đã lấy
    retriever_bm25 = BM25Retriever.from_documents(top_docs)
    retriever_bm25.k = 4

    ## Kết hợp hai retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_faiss, retriever_bm25],
        weights=[0.7, 0.3]
    )

    return ensemble_retriever


def get_llm(model_choice):
    """
    Trả về LLM object của LangChain
    """
    return ChatOllama(
        model=model_choice,
        temperature=0.001,
        base_url="http://localhost:11434",
        device="cuda"  # Chạy trên gpu
    )

def get_answer(chat_history, user_question, retriever, model_choice):
    # Lấy các tài liệu liên quan
    docs = retriever.invoke(user_question)
    context = "\n".join(doc.page_content for doc in docs)

    # Tạo ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là chuyên gia về AI, trả lời bằng Tiếng Việt. Tên bạn là NVP-Chatbot.
    Chỉ trả lời dựa trên thông tin có trong tài liệu và lịch sử hội thoại đã được cung cấp bên dưới.
    Tuyệt đối KHÔNG được suy đoán, bịa đặt hoặc đưa thông tin ngoài phạm vi tài liệu.
    Nếu câu hỏi không thể trả lời từ các thông tin này, hãy trả lời đúng nguyên văn: "Tôi không chắc chắn".
    Khi trả lời, hãy phân tích và trình bày rõ ràng, có thể so sánh hoặc đánh giá nếu dữ liệu cho phép.

    --- Lịch sử hội thoại ---
    {chat_history}

    --- Tài liệu hỗ trợ ---
    {context}
    """
    ),
    ("human", "{question}")
    ])


    # Format prompt với dữ liệu thực tế
    messages = prompt.format_messages(
        chat_history=chat_history,
        context=context,
        question=user_question
    )
    print("📏 Context length:", len(context), "ký tự")

    # Gọi model
    llm = get_llm(model_choice)

    # Dùng streaming
    full_response = ""
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content  # Trả về từng phần text
            full_response += chunk.content

    # Không dùng cho streaming
    # response = llm.invoke(messages)
    # return response.content  # Trả về toàn bộ câu trả lời
