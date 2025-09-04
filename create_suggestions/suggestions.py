# import os
# from langchain.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # Hàm sinh ra gợi ý tiếp theo
# def generate_suggestions_from_docs(query, retrieved_docs, model_choice):
#     """
#     Sinh ra 1 câu gợi ý dựa vào tài liệu đã truy xuất (retriever).
#     """
#     retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

#     llm = ChatGoogleGenerativeAI(
#         model= model_choice,
#         temperature=0,
#         google_api_key=GOOGLE_API_KEY,
#         model_kwargs={"streaming": True},
#     )

#     suggestion_prompt = PromptTemplate(
#         input_variables=["query", "retrieved_text"],
#         template="""
#         Người dùng vừa hỏi: "{query}"

#         Đây là một số thông tin liên quan bạn tìm được:
#         {retrieved_text}

#         Dựa vào tài liệu trên, hãy đưa ra 1 gợi ý câu hỏi tiếp theo 
#         (ngắn gọn, tự nhiên, liên quan trực tiếp).
#         """
#     )

#     chain = suggestion_prompt | llm |
#     suggestions = chain.invoke({"query": query, "retrieved_text": retrieved_text}).content

#     return suggestions