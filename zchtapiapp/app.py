import json
from dotenv import load_dotenv
import os
from utils import EndpointRetriever, call_api, ChatGroq, ChatPromptTemplate, get_updated_payload_and_url, TaskChatAssistant

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", temperature=0)

def fetch_and_format_categories(crt_pyld_url):
    # Define the default category URL inside the function
    category_url = "https://amt-gcp-dev.soham.ai/inventory-task/v1/get_task_categories"
    
    if "category_id" in crt_pyld_url[0]:
        entity_id = os.getenv("entity_id", "")
        crt_pyld_url[0]["entity_id"] = entity_id

        # Call API with the default category URL and entity_id
        ctgry_Api_response = call_api(category_url, {"entity_id": entity_id})
        categories = ctgry_Api_response.get("categories", [])

        # Format the categories for display
        formatted_categories = [
            f"{index + 1}. ID: {category.get('id', 'N/A')}, Name: {category.get('name', 'N/A')}"
            for index, category in enumerate(categories)
        ]
        return "\n".join(formatted_categories)
    
    return "No categories found."


def get_task_sub_categories(task_category_id):
    entity_id = os.getenv("entity_id", "")
    
    # Prepare parameters for the API request
    url = "https://amt-gcp-dev.soham.ai/inventory-task/v1/get_task_sub_categories"
    params = {
        "task_category_id": task_category_id,
        "entity_id": entity_id
    }
    response = call_api(url, params)
    
    # Check if the response contains categories and format them with numeric indexing
    if 'categories' in response:
        categories = response['categories']
        formatted_categories = [
            f"{index + 1}. ID: {category['id']}, Name: {category['name']}"
            for index, category in enumerate(categories)
        ]
        return "\n".join(formatted_categories)
    else:
        return "No categories found or an error occurred."

class TaskQAssistant:
    def __init__(self, llm):
        self.llm = llm

    def generate_task_questions(self):
        task_data_template = {
            "title": "",
            "entity_id": "",
            "requested_by": "",
            "message": "",
            "category_id": "",
            "sub_category_id": ""
        }
        task_data_str = str(task_data_template)

        chat_prompt = ChatPromptTemplate.from_template(
            "You are an AI assistant helping to gather task details. "
            "The following fields are missing and need to be filled:\n\n"
            "{context}\n\n"
            "Please ask clear, concise questions to gather the necessary details for each field.\n"
            "Each question should directly request the missing information.\n\n"
            "Your response should be a list of numbered questions, one per line, for the user to answer."
        )

        conversation_chain = chat_prompt | self.llm
        chat_response = conversation_chain.invoke({"context": task_data_str})

        task_questions = chat_response.content
        cleaned_text = task_questions.replace(
            "Assistant: Here are the questions to gather the necessary details:\n\n", ""
        ).replace(
            "Please answer these questions to fill in the missing fields!", ""
        ).strip()

        questions_list = [
            question.strip() + '?' for question in cleaned_text.split('\n') if question.strip()
        ]
        questions_list = [q.replace("??", "?") for q in questions_list]

        return task_questions, questions_list

    def ask_for_responses(self, questions_list, crt_pyld_url):
        keys = ['title', 'entity_id', 'requested_by', 'message', 'category_id', 'sub_category_id']
        responses = {}

        # Pre-fill entity_id automatically
        entity_id = os.getenv("entity_id", "")
        responses['entity_id'] = entity_id

        categories_list = fetch_and_format_categories(crt_pyld_url)

        for index, question in enumerate(questions_list[1:], start=0):
            if "category" in question.lower() and responses.get('category_id') is None:
                print(f"Fetching categories...")
                print(f"Available categories:\n{categories_list}")
            
            if keys[index] == 'entity_id':  # Skip entity_id question
                continue

            # Ask the category question
            response = input(f"{question} ")
            responses[keys[index]] = response

            # If category_id is provided, get subcategories
            if keys[index] == 'category_id' and responses.get('category_id'):
                category_id = responses['category_id']
                print(f"Fetching subcategories for category ID: {category_id}...")
                sub_categories = get_task_sub_categories(category_id)

                print(f"Available subcategories for category ID {category_id}:\n{sub_categories}")

        return responses

def create_task_payload(llm, query):
    crt_pyld_url = get_updated_payload_and_url(query)
    assistant = TaskQAssistant(llm)
    task_questions, questions_list = assistant.generate_task_questions()
    crtT_pyld = assistant.ask_for_responses(questions_list, crt_pyld_url)

    return crtT_pyld

# Main function
# def main():
#     while True:  # Infinite loop to keep asking for queries
#         query = input("Please enter the query (or type 'exit' to restart): ")  
        
#         if query.lower() == "exit":
#             print("Restarting the process...\n")
#             continue  # Restart the loop instead of exiting

#         crt_pyld_url = get_updated_payload_and_url(query)
#         payload = crt_pyld_url[0]
#         endpoint_url = crt_pyld_url[1]

#         print("\nUpdated Payload:", payload)
#         print("\nEndpoint URL:", endpoint_url)

#         if "create_task" in endpoint_url:
#             crtT_payload = create_task_payload(llm, query)
#             print("\nCreated Task Payload:", crtT_payload)
#         else:
#             apires = call_api(endpoint_url, payload)
#             task_chat_assistant = TaskChatAssistant(task_data=apires, groq_api_key=groq_api_key)
#             task_chat_assistant.chat_loop()

# # Execute main function
# if __name__ == "__main__":
#     main()


# assistant = TaskQAssistant(llm)

# task_questions, questions_list = assistant.generate_task_questions()

# # Print the results
# print("Task Questions (Full Response):")
# print(task_questions)

# print("\nTask Questions (List):")
# for question in questions_list:
#     print(question)
