import os
import glob
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", temperature=0)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

FAISS_INDEX_PATH = "data/faiss_index"
vectors = None

# 🔹 Simple answer prompt
prompt = ChatPromptTemplate.from_template(
    "Using the provided context, answer the user's query in a simple and concise manner.\n\n"
    "Context: {context}\n\nQuery: {input}\n\nAnswer:"
)

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

        # Creating vector embeddings
        with open(self.file_path, 'r', encoding="utf-8") as file:
            text = file.read()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
        chunks = text_splitter.split_text(text)

        documents = [Document(page_content=chunk) for chunk in chunks]
        self.vectors = FAISS.from_documents(documents, embeddings)
        self.vectors.save_local(FAISS_INDEX_PATH)

        return {"status": "FAISS index successfully created."}  # No unnecessary print statements

    def retrieval(self, query):
        """Retrieves the most relevant answer based on the user's query."""
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
            vector_status = self.vector_embedding()  # Initialize vectors if not loaded
        else:
            vector_status = {"status": "FAISS index already loaded."}  # If FAISS is already loaded
        
        # Process the query directly
        answer = self.retrieval(query)

        end_time = time.time()  # End time after query processing
        response_time = round(end_time - start_time, 2)  # Calculate response time

        return {
            "vector_status": vector_status,  # Include vector status in response
            "query": query,
            "Model Response": answer,
            "response_time": f"{response_time} seconds"  # Include response time
        }

# 🔹 Provide the path to your text file
text_file_path = "data/document.txt"  # Change this to your actual text file

# 🔹 Create an instance of the retriever
retriever = EndpointRetriever(text_file_path)

# 🔹 Run vector embedding (only once)
retriever.vector_embedding()

# 🔹 Set query directly
query = "how to setup cellular service"

# 🔹 Process query
response = retriever.process_query(query)

# 🔹 Print the response
print("\n--- Query Response ---")
print(response)
