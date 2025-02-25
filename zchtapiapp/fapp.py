#terminal based =================================================================================================

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import os
# from utils import call_api, get_updated_payload_and_url, TaskChatAssistant,ChatGroq
# import uvicorn
# from app import create_task_payload
# from dotenv import load_dotenv

# # Initialize FastAPI app
# app = FastAPI()

# load_dotenv()

# groq_api_key = os.getenv('GROQ_API_KEY')
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", temperature=0)
# # Load environment variables
# entity_id = os.getenv("entity_id", "")
# groq_api_key = os.getenv('GROQ_API_KEY', "")

# # Request model for query input
# class QueryRequest(BaseModel):
#     query: str

# @app.post("/query")
# def handle_query(request: QueryRequest):
#     """API endpoint to process a query and return the response."""
#     query = request.query

#     try:
#         # Get the updated payload and endpoint URL
#         crt_pyld_url = get_updated_payload_and_url(query)
#         payload = crt_pyld_url[0]
#         endpoint_url = crt_pyld_url[1]
#         print("payload : ",payload)
#         print("endpoint_url : ",endpoint_url)

#         # If the endpoint is for task creation, generate the task payload
#         if "create_task" in endpoint_url:
#             response_data = create_task_payload(llm,query)
        # else:
        #     # Call the external API with the given payload
        #     response_data = call_api(endpoint_url, payload)

        #     # Use TaskChatAssistant to process the response
        #     task_chat_assistant = TaskChatAssistant(task_data=response_data, groq_api_key=groq_api_key)
        #     response_data = task_chat_assistant.chat_loop()

#         return {"status": "success", "data": response_data}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# # Run the FastAPI app if executed directly
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

#for both crete and none create task api =================================================================================================

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import os
# from utils import call_api, get_updated_payload_and_url, ChatGroq
# from dotenv import load_dotenv
# from app import TaskQAssistant, TaskChatAssistant

# # Initialize FastAPI app
# app = FastAPI()

# load_dotenv()

# groq_api_key = os.getenv('GROQ_API_KEY')
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", temperature=0)

# # Store user progress (temporary storage for demo; use DB for production)
# user_sessions = {}
# chat_sessions = {}  # Stores chat sessions for ongoing queries

# # Get default entity_id from environment
# DEFAULT_ENTITY_ID = os.getenv("entity_id", "")

# # Request models
# class QueryRequest(BaseModel):
#     query: str

# class AnswerRequest(BaseModel):
#     session_id: str
#     user_query: str

# class ChatRequest(BaseModel):
#     session_id: str
#     user_query: str

# @app.post("/start-task")
# def start_task(request: QueryRequest):
#     query = request.query
#     try:
#         crt_pyld_url = get_updated_payload_and_url(query)
#         payload, endpoint_url = crt_pyld_url
#         print("payload : ", payload)
#         print("endpoint_url : ", endpoint_url)
        

#         if "create_task" in endpoint_url:
#             assistant = TaskQAssistant(llm)
#             _, questions_list = assistant.generate_task_questions()
#             filtered_questions = [q for i, q in enumerate(questions_list[1:], start=1) if i != 2]
#             user_sessions[query] = {
#                 "questions": filtered_questions,
#                 "answers": {"entity_id": DEFAULT_ENTITY_ID},
#                 "current_index": 0,
#                 "payload": payload
#             }
#             return {"status": "waiting_for_answer", "question": user_sessions[query]["questions"][0]}
        
#         # If not a task creation request, start a chat session
#         response_data = call_api(endpoint_url, payload)
#         session_id = str(len(user_sessions) + 1)  # Unique session ID

#         user_sessions[session_id] = {
#             "task_data": response_data
#         }

#         return {"status": "chat_started", "session_id": session_id, "message": "Chat session started.","payload": payload, "endpoint": endpoint_url}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# from typing import Optional

# class AnswerRequest(BaseModel):
#     session_id: Optional[str] = None  # Used for chat
#     user_query: Optional[str] = None  # Used for chat queries
#     query: Optional[str] = None  # Used for task creation
#     answer: Optional[str] = None  # Used for task creation

# @app.post("/next-question")
# def next_question(request: AnswerRequest):
#     """Handles user answers one by one and returns the next question."""
    
#     if request.session_id and request.user_query:
#         # This is a chat request
#         session_id = request.session_id
#         user_query = request.user_query

#         if session_id not in user_sessions:
#             return {"status": "error", "message": "Session not found. Start a session first."}

#         session = user_sessions[session_id]
#         task_chat_assistant = TaskChatAssistant(task_data=session["task_data"], groq_api_key=groq_api_key)

#         assistant_response = task_chat_assistant.chat_loop(user_query)
#         return {"status": "success", "response": assistant_response}
    
#     elif request.query and request.answer:
#         # This is a task creation request
#         query = request.query
#         answer = request.answer

#         if query not in user_sessions:
#             return {"status": "error", "message": "Session not found. Please restart."}

#         session = user_sessions[query]

#         # Define the correct order of expected keys (excluding entity_id from user input)
#         keys = ['title', 'requested_by', 'message', 'category_id', 'sub_category_id']

#         # Store user responses in order, skipping entity_id (it's set by default)
#         current_index = session["current_index"]

#         if current_index < len(keys):  # Ensure we don't go out of index bounds
#             session["answers"][keys[current_index]] = answer

#         # Move to the next question
#         session["current_index"] += 1

#         if session["current_index"] < len(session["questions"]):
#             next_q = session["questions"][session["current_index"]]
#             return {"status": "waiting_for_answer", "question": next_q}
#         else:
#             # All questions answered, create final task payload
#             final_payload = session["payload"]
#             final_payload.update(session["answers"])  # Merge answers with payload

#             # Ensure entity_id remains default
#             final_payload["entity_id"] = DEFAULT_ENTITY_ID

#             # Cleanup session
#             del user_sessions[query]

#             return {
#                 "status": "success",
#                 "task_payload": final_payload
#             }
    
#     return {"status": "error", "message": "Invalid request format."}


# @app.post("/chat")
# def chat(request: AnswerRequest):
#     session_id = request.session_id
#     user_query = request.user_query  # Extract user query from request

#     if session_id not in user_sessions:
#         return {"status": "error", "message": "Session not found. Start a session first."}

#     session = user_sessions[session_id]
#     task_chat_assistant = TaskChatAssistant(task_data=session["task_data"], groq_api_key=groq_api_key)

#     # Call chat_loop with user_query to get a response
#     assistant_response = task_chat_assistant.chat_loop(user_query)

#     return {"status": "success", "response": assistant_response}


# # Run FastAPI app
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# with single endpoints=================================================================================================

from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
import os
from typing import Optional, Dict
from utils import call_api, get_updated_payload_and_url, ChatGroq, get_task_sub_categories
from dotenv import load_dotenv
from app import TaskQAssistant, TaskChatAssistant
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from create_task_api_cl import create_task_api_cl
from utils import EndpointRetriever
load_dotenv()
# from pyngrok import ngrok

# Initialize FastAPI app
app = FastAPI()

# ngrokToken = os.getenv('NGROK_TOKEN')
# ngrok.set_auth_token(ngrokToken)
# public_url = ngrok.connect(5000).public_url
# print(f"Access the global link: {public_url}")

app.mount("/Users/macbook/Desktop/headless_app_chat_with_api/zchtapiapp/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
DEFAULT_ENTITY_ID = os.getenv("entity_id", "")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", temperature=0)

# Global variable to hold the retriever instance
endpoint_retriever = None  

@app.on_event("startup")
async def startup_event():
    global endpoint_retriever
    file_path = "/Users/macbook/Desktop/headless_app_chat_with_api/zchtapiapp/data/api.txt"  
    if endpoint_retriever is None:  # Prevents re-initialization
        endpoint_retriever = EndpointRetriever(file_path)
        endpoint_retriever.vector_embedding()  # Load once at startup

@app.on_event("shutdown")
async def shutdown_event():
    global endpoint_retriever
    endpoint_retriever = None  # Cleanup on shutdown


# Store user progress (temporary storage, use DB for production)
user_sessions: Dict[str, Dict] = {}

# Request model (Single Input Field)
class ChatRequest(BaseModel):
    message_id: Optional[str] = None
    user_input: str  # Single field for both chat & task creation


@app.get("/")
def get_chat_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

import uuid
@app.post("/chat")
def chat(request: ChatRequest):
    user_input = request.user_input.strip()
    user_id = "qwertyuioaasghjk"
    message_id = str(uuid.uuid4())
    # Check if user is responding to a task creation session
    for query, session in user_sessions.items():
        if "questions" in session and "current_index" in session:
            current_index = session["current_index"]
            keys = ['title', 'requested_by', 'message', 'category_id', 'sub_category_id']

            if current_index < len(keys):
                session["answers"][keys[current_index]] = user_input
                session["current_index"] += 1
            
            if session["current_index"] < len(session["questions"]):
                next_q = session["questions"][session["current_index"]]
                return {"status": "waiting_for_answer", "question": next_q, "user_id": user_id}
            else:
                # All questions answered, finalize task
                final_payload = session["payload"]
                final_payload.update(session["answers"])
                final_payload["entity_id"] = DEFAULT_ENTITY_ID
                task_url = session["endpoint_url"]

                print("\n session_payload : ", final_payload)
                print("\n session_url : ", task_url)
                
                del user_sessions[query]  # Cleanup session
                return {"status": "success", "task_payload": final_payload, "task_url": task_url,"user_id": user_id}

    # If user is providing a new query (not answering task-related questions)
    try:
        payload, endpoint_url = get_updated_payload_and_url(user_input)

        if "create_task" in endpoint_url:
            assistant = TaskQAssistant(llm)
            _, questions_list = assistant.generate_task_questions()
            filtered_questions = [q for i, q in enumerate(questions_list[1:], start=1) if i != 2]

            user_sessions[user_input] = {
                "questions": filtered_questions,
                "answers": {"entity_id": DEFAULT_ENTITY_ID},
                "current_index": 0,
                "payload": payload,
                "endpoint_url": endpoint_url
            }
            return {"status": "waiting_for_answer", "message_id": message_id, "question": filtered_questions[0], "payload": payload, "endpoint_url": endpoint_url,"user_id": user_id}

        # General chat-based request
        print("\n endpoint_url : ", endpoint_url)
        print("\n payload : ", payload)
        
        response_data = call_api(endpoint_url, payload)
        # message_id = str(len(user_sessions) + 1)
        user_sessions[message_id] = {"task_data": response_data}
        task_chat_assistant = TaskChatAssistant(task_data=response_data, groq_api_key=groq_api_key)
        assistant_response = task_chat_assistant.chat_loop(user_input)

        return {"status": "success", "message_id": message_id, "response": assistant_response, "user_id": user_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fetch_categories")
async def fetch_categories():
    category_url = "https://amt-gcp-dev.soham.ai/inventory-task/v1/get_task_categories"
    payload = {"entity_id": DEFAULT_ENTITY_ID}
    
    try:
        response = call_api(category_url, payload)
        categories = response.get("categories", [])
        if not categories:
            return {"categories": ["No categories found."]}

        formatted_categories = [f"ID: {category.get('id', 'N/A')}, Name: {category.get('name', 'N/A')}" for category in categories]
        return {"categories": formatted_categories}

    except Exception as e:
        return {"categories": [f"Error fetching categories: {str(e)}"]}


@app.get("/fetch_subcategories/{category_id}")
async def fetch_subcategories(category_id: int):
    return {"subcategories": get_task_sub_categories(category_id)}


@app.post("/create-task")
def create_task(request_payload: Dict):
    """
    Endpoint to create a task dynamically.
    Requires payload to be sent as JSON in the request body.
    """
    endpoint_url = "https://amt-gcp-dev.soham.ai/inventory-task/v1/create_task"
    
    try:
        response = create_task_api_cl(endpoint_url, request_payload)
        print(response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
