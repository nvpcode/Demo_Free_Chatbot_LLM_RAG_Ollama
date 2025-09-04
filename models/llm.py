from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from retriever.retriever import show_retrieved_chunks
from rerank.reranker import show_rerank_chunks

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

def get_answer_from_llm(chat_history, user_question, retriever_obj, rerank_obj, model_choice):
    # Lấy các tài liệu liên quan
    retrieved_docs = show_retrieved_chunks(retriever_obj, user_question)
    reranked_docs = show_rerank_chunks(rerank_obj, user_question)   
    context = "\n".join(doc.page_content for doc in reranked_docs)
    
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
    print("📏 Context length:", len(context), "ký tự\n")

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

    

def get_answer_from_llm_2(chat_history, user_question, retriever_obj, rerank_obj, model_choice):
    # Lấy các tài liệu liên quan
    retrieved_docs = show_retrieved_chunks(retriever_obj, user_question)
    reranked_docs = show_rerank_chunks(rerank_obj, user_question)   
    context = "\n".join(doc.page_content for doc in reranked_docs)
    
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

    # Gọi model
    llm = get_llm(model_choice)

    # Không dùng cho streaming
    response = llm.invoke(messages)

    return response.content, reranked_docs  
