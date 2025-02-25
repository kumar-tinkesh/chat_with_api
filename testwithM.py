# from groq import Groq
# from langchain.chains import LLMChain
# from langchain_core.prompts import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     MessagesPlaceholder,
# )
# from langchain_core.messages import SystemMessage
# from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# from langchain_groq import ChatGroq

# # def chat_with_session():
# #     """
# #     Terminal-based chatbot using Groq and LangChain.
# #     """
# #     # Get Groq API key
# #     groq_api_key = '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'  # Replace 'your_api' with your actual API key

# #     # User customization options
# #     system_prompt = "You are a ticketing system"
# #     model = "llama3-8b-8192"
# #     conversational_memory_length = 10
    
# #     memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)
# #     chat_history = []

# #     # Initialize Groq LangChain chat object
# #     groq_chat = ChatGroq(
# #         groq_api_key=groq_api_key,
# #         model_name=model
# #     )

# #     while True:
# #         user_question = input("You: ")
# #         if user_question.lower() in ["exit", "quit", "bye"]:
# #             print("Chatbot: Goodbye!")
# #             break

# #         # Construct a chat prompt template
# #         prompt = ChatPromptTemplate.from_messages(
# #             [
# #                 SystemMessage(content=system_prompt),
# #                 MessagesPlaceholder(variable_name="chat_history"),
# #                 HumanMessagePromptTemplate.from_template("{human_input}"),
# #             ]
# #         )

# #         # Create a conversation chain
# #         conversation = LLMChain(
# #             llm=groq_chat,
# #             prompt=prompt,
# #             verbose=False,
# #             memory=memory,
# #         )

# #         # Generate chatbot response
# #         response = conversation.predict(human_input=user_question)
# #         chat_history.append({'human': user_question, 'AI': response})
# #         print("Chatbot:", response)

# # # if __name__ == "__main__":
# # #     chat_with_session()

# # ans = chat_with_session()


# def chat_with_session(task_response=None):
#     """
#     Terminal-based chatbot using Groq and LangChain.
#     The chatbot will use task response as context.
#     """
#     # Get Groq API key
#     groq_api_key = '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'  # Replace 'your_api' with your actual API key

#     # User customization options
#     system_prompt = (
#             "You are a ticketing system designed to manage and process assigned tasks. "
#             "Your goal is to respond to inquiries regarding tasks and provide concise, accurate information. "
#             "Follow these guidelines when responding to user queries:\n\n"
#             "Remember, if no relevant task data is available for a query, provide a response such as 'No tasks found in this category.'"
#     )
#     model = "llama3-8b-8192"
#     conversational_memory_length = 10
    
#     memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)
#     chat_history = []

#     # Include task response in initial system memory if available
#     if task_response:
#         task_summary = f"Task info: {task_response}"
#         chat_history.append({'system': task_summary})

#     # Initialize Groq LangChain chat object
#     groq_chat = ChatGroq(
#         groq_api_key=groq_api_key,
#         model_name=model
#     )

#     while True:
#         user_question = input("You: ")
#         if user_question.lower() in ["exit", "quit", "bye"]:
#             print("Chatbot: Goodbye!")
#             break

#         # Construct a chat prompt template
#         prompt = ChatPromptTemplate.from_messages(
#             [
#                 SystemMessage(content=system_prompt),
#                 MessagesPlaceholder(variable_name="chat_history"),
#                 HumanMessagePromptTemplate.from_template("{human_input}"),
#             ]
#         )

#         # Create a conversation chain
#         conversation = LLMChain(
#             llm=groq_chat,
#             prompt=prompt,
#             verbose=False,
#             memory=memory,
#         )

#         # Generate chatbot response
#         response = conversation.predict(human_input=user_question)
#         chat_history.append({'human': user_question, 'AI': response})
#         print("Chatbot:", response)

# # Start the chatbot session with the task response passed as context
# # chat_with_session(task_response)


