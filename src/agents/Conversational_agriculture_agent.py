from langchain_ollama import ChatOllama
from src.config.model_config import (
    OLLAMA_BASE_URL, 
    OLLAMA_MODEL, 
    TEMPERATURE, 
    MAX_TOKENS, 
    CONVERSATIONAL_AGENT_PROMPT
)

def create_conversational_agent():
    """Create the conversational agriculture agent without tools"""
    
    # Initialize the model
    model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    # Return the model directly
    return model