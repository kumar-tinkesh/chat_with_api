import os
import glob
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from putils import extract_and_format_key_value_pairs_from_user_prompt

# Load environment variables
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", temperature=0)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

FAISS_INDEX_PATH = "data/faiss_index"
vectors = None

prompt = ChatPromptTemplate.from_template(
    "Retrieve the most relevant endpoint(s) from the provided context based on the user's query. "
    "Ensure the response strictly follows this format without deviation: \n\n"
    "{{\n"
    "  \"endpoint_url\": \"<URL>\",\n"
    "  \"payload\": {{\n"
    "    \"field1\": \"<value>\",\n"
    "    \"field2\": \"<value>\",\n"
    "    \"field3\": \"<value>\"\n"
    "    // Add additional fields as required\n"
    "  }}\n"
    "}}\n\n"
    "If a field value is not provided, leave it blank (e.g., \"field1\": \"\"). "
    "Replace <URL> with the actual endpoint URL and <value> with default or placeholder values for each field. "
    "Do not include additional text, explanations, or comments outside this JSON format.\n\n"
    "Context: {context}\n\nQuery: {input}\n"
)
import time

class EndpointRetriever:
    _instance = None  # Singleton instance

    def __new__(cls, file_path):
        if cls._instance is None:
            cls._instance = super(EndpointRetriever, cls).__new__(cls)
            cls._instance.file_path = file_path
            cls._instance.vectors = None
        return cls._instance

    def vector_embedding(self):
        """Creates or loads FAISS vector embeddings."""
        embeddings = HuggingFaceEmbeddings()
        
        if self.vectors is not None:
            print("FAISS index already loaded, skipping reload.")
            return {"status": "FAISS index already loaded."}

        if os.path.exists(FAISS_INDEX_PATH) and glob.glob(f"{FAISS_INDEX_PATH}/*"):
            print("Loading existing FAISS index...")
            self.vectors = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            return {"status": "Loaded existing FAISS index."}

        print("Creating new vector embeddings...")
        with open(self.file_path, 'r') as file:
            text = file.read()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
        chunks = text_splitter.split_text(text)

        documents = [Document(page_content=chunk) for chunk in chunks]
        self.vectors = FAISS.from_documents(documents, embeddings)
        self.vectors.save_local(FAISS_INDEX_PATH)

        print("Vector embeddings created and saved.")
        return {"status": "Vector embeddings created and saved."}


    def retrieval(self, query):
        """Retrieves the most relevant endpoint based on the user's query."""
        if not self.vectors:
            print("Vector store not initialized. Please run the embedding function first.")
            return None

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = self.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({'context': 'Your context here', 'input': query})

        try:
            return response['answer']
        except Exception as e:
            print(f"Error: {e}")
            return None


    def process_query(self, query):
        """Main function that initializes embeddings and processes the query."""
        start_time = time.time()  # Start time for measuring response time
        vector_status = None
        if self.vectors is None:
            print("Reloading FAISS index inside process_query (this should not happen if startup loaded it)")
            vector_status = self.vector_embedding()  # Get the vector embedding status
        else:
            vector_status = {"status": "FAISS index already loaded."}  # If FAISS is already loaded
        # Process the query
        f, Q = extract_and_format_key_value_pairs_from_user_prompt(query)
        Query_Intent = Q.get('query_intent')
        answer = self.retrieval(Query_Intent)

        end_time = time.time()  # End time after query processing
        response_time = end_time - start_time  # Calculate response time

        return {
            "vector_status": vector_status,  # Include vector status in response
            "query": query,
            "Model Response1": answer,
            "response_time": response_time  # Include response time in seconds
        }

# retriever = EndpointRetriever(file_path="data/api.txt")
# pld_url = retriever.process_query()

class TaskChatAssistant:
    def __init__(self, task_data, groq_api_key, model_name="Llama3-8b-8192", temperature=0):
        self.task_data = task_data
        self.groq_api_key = groq_api_key
        self.llm = ChatGroq(groq_api_key=self.groq_api_key, model_name=model_name, temperature=temperature)

    def chat_with_task_data(self, user_message):

        task_data_str = str(self.task_data)  # Convert task data to string (or format as JSON if needed)
        
        if len(task_data_str) > 6000:
            print("Your content is too long. Please shorten the task data or provide a more specific query.")
            return "The task data is too lengthy to process. Please provide a more specific query."

        # Creating the prompt template with the task data context
        chat_prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant.if The following context provides detailed information related to various tasks:\n\n"
            "{context}\n\n"
            "User's Query: {input}\n\n"
            "Your task is to retrieve the most relevant data points related to the query. "
            "Provide the answer in a clear, concise, and informative manner. Avoid unnecessary details.\n\n"
            "Response:"
        )
        
        # Chain the prompt with the model for conversation
        conversation_chain = chat_prompt | self.llm  # Chaining the prompt with the model
        
        # Invoke the chain with the task data and user message
        chat_response = conversation_chain.invoke({
            "context": task_data_str,  # Correct variable name is 'context'
            "input": user_message  # Correct variable name is 'input'
        })
        
        # Extract and return the assistant's response
        assistant_message = chat_response.content
        return assistant_message


    # def chat_loop(self):

    #     print("Welcome! You can ask me about the tasks. Type 'exit' to quit the conversation.")
        
    #     while True:
    #         # Get user input
    #         user_message = input("You: ")
            
    #         # Exit the loop if the user types 'exit'
    #         if user_message.lower() == 'exit':
    #             print("Goodbye!")
    #             break
    #         responses = []
    #         # Get the assistant's response
    #         response = self.chat_with_task_data(user_message)
    #         responses.append(response)
    #         # Print the assistant's response
    #         print(f"Assistant: {response}")
    #     return responses
    
    def chat_loop(self, user_query=None):
        """Handles a single query if provided, otherwise enters interactive mode."""
        if user_query:
            return self.chat_with_task_data(user_query)

        print("Welcome! You can ask me about the tasks. Type 'exit' to quit the conversation.")
        while True:
            user_message = input("You: ")
            if user_message.lower() == 'exit':
                print("Goodbye!")
                break
            response = self.chat_with_task_data(user_message)
            print(f"Assistant: {response}")




import requests
def call_api(endpoint_url, payload):
    """Send an API request to the specified URL with the given payload as form-data."""
    BToken = os.getenv('BearerToken')  # Get Bearer token from environment variable
    headers = {
        "Authorization": f"Bearer {BToken}"
    }

    try:
        # print(f"Sending API request to: {endpoint_url}")
        
        # Ensure the payload is a dictionary before proceeding
        if isinstance(payload, dict):
            form_data = {key: str(value) for key, value in payload.items()}
            
            response = requests.post(endpoint_url, headers=headers, data=form_data)
            
            if response.status_code == 200:
                print("API call successful!")
                return response.json()
            else:
                print(f"API call failed with status code {response.status_code}.")
                print("Response:", response.text)
                return {"error": "API call failed", "status_code": response.status_code, "response": response.text}
        else:
            print("Invalid payload. Must be a dictionary.")
            return {"error": "Invalid payload", "message": "Payload must be a dictionary."}
    
    except Exception as e:
        print(f"Error: API call failed with exception: {e}")
        return {"error": "Exception occurred", "message": str(e)}

import json
def get_updated_payload_and_url(query):
    """
    Retrieves the model response, updates the payload based on the endpoint URL,
    and returns the updated payload and URL.
    """
    # Create an instance of EndpointRetriever with the file path
    retriever = EndpointRetriever(file_path="data/api.txt")
    # query = "assigned task"
    print("retriever: ", retriever)
    model_response = retriever.process_query(query)
    # print(model_response)
    # Get the model response JSON (this will be a dictionary)
    model_response_json = model_response.get("Model Response1", {})
    
    # Check if the model response is already a dictionary
    if isinstance(model_response_json, dict):
        model_response_dict = model_response_json
    else:
        try:
            # If it's not a dictionary, parse it as JSON
            model_response_dict = json.loads(model_response_json)
        except json.JSONDecodeError:
            print("Error: Invalid JSON response received.")
            return {}, ""  # Return empty payload and URL if parsing fails
    
    # Get the payload and endpoint_url from the model response
    payload = model_response_dict.get("payload", {})
    endpoint_url = model_response_dict.get("endpoint_url", "")
    # print("endpoint_url:", endpoint_url)
    # print("payload:", payload)
    
    if "create_task" in endpoint_url:
        return payload, endpoint_url

    # Update payload based on the endpoint URL
    if "get_assigned_tickets" in endpoint_url:
        assignee_id = os.getenv("assignee_id", "")
        entity_id = os.getenv("entity_id", "")
        payload["assignee_id"] = assignee_id
        payload["entity_id"] = entity_id
        payload["page"] = 1
        payload["limit"] = 10

    elif "get_requested_tickets" in endpoint_url:
        requester_id = os.getenv("requested_by", "")
        entity_id = os.getenv("entity_id", "")
        payload["requested_by"] = requester_id
        payload["entity_id"] = entity_id
        payload["page"] = 1
        payload["limit"] = 10

    elif "get_all_tickets" in endpoint_url:
        entity_id = os.getenv("entity_id", "")
        payload["entity_id"] = entity_id

    elif "get_ticket_categories" in endpoint_url:
        entity_id = os.getenv("entity_id", "")
        payload["entity_id"] = entity_id

    return payload, endpoint_url


def get_task_sub_categories(category_id):
    entity_id = os.getenv("entity_id", "")
    url = "https://amt-gcp-dev.soham.ai/inventory-task/v1/get_task_sub_categories"
    payload = {"task_category_id": category_id, "entity_id": entity_id}
    
    try:
        response = call_api(url, payload)

        if response and 'categories' in response:
            return [
                {"id": category["id"], "name": category["name"]}
                for index, category in enumerate(response['categories'])
            ] if response['categories'] else []
        else:
            return []
    except Exception as e:
        return []


# fv = get_task_sub_categories(17)
# print(fv)