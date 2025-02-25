import re
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

def extract_and_format_key_value_pairs_from_user_prompt(query):
    # Load API key
    load_dotenv()
    groq_api_key = os.getenv('GROQ_API_KEY')
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", temperature=0)

    # Define flexible intent mapping using regex patterns
    intent_mapping = {
        r"\bassigned\b": "assigned_tickets",
        r"\brequest(ed)?\b": "requested_tickets",
        r"\b(all tickets|show tickets)\b": "get_all_tickets",
        r"\b(ticket categor(y|ies))\b": "get_ticket_categories",
        r"\b(create|make|add) (a )?(new )?(task|ticket|tickets)\b": "create_task",
    }

    # Extract query intent
    detected_intent = "general_inquiry"  # Default intent
    for pattern, intent in intent_mapping.items():
        if re.search(pattern, query, re.IGNORECASE):
            detected_intent = intent
            break

    # Remove matched intent words from the query
    additional_details = re.sub(r'\b(?:' + '|'.join(intent_mapping.keys()) + r')\b', '', query, flags=re.IGNORECASE).strip()

    # **Prompt Template**
    prompt_template = ChatPromptTemplate.from_template(
        """You are an advanced AI specializing in extracting structured key-value pairs from user queries. 
        
        **Task:**  
        Extract key-value pairs from the given input. Ensure:  
        1. The **query_intent** is identified based on keywords (e.g., "assigned" → `assigned_tickets`, "requested" → `requested_tickets`).  
        2. Any additional details are extracted as separate key-value pairs.  
        3. The response follows strict JSON format, using underscores for keys.

        ---

        **User Query:**  
        "{query}"

        """
    )

    # Format the prompt with extracted values
    formatted_prompt = prompt_template.format(query=query, query_intent=detected_intent, additional_details=additional_details)
    response = llm.invoke(formatted_prompt)

    # Extract response content
    output = response.content.strip()

    # Ensure output contains query intent
    if "query_intent" not in output:
        output = f"{{\"query_intent\": \"{detected_intent}\"}},\n" + output

    # Extract key-value pairs using regex
    key_value_list = re.findall(r'"([^"]+)":\s*"([^"]+)"', output)

    # Convert key-value pairs into a dictionary
    key_value_dict = {key: value for key, value in key_value_list}

    return output, key_value_dict

# query = "get requested tickets"
# ns, key_value_dict = extract_and_format_key_value_pairs_from_user_prompt(query)
# print(key_value_dict)