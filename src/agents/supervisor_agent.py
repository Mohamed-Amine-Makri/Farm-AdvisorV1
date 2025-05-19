from langchain_ollama import ChatOllama
from src.config.model_config import (
    OLLAMA_BASE_URL, 
    OLLAMA_MODEL, 
    TEMPERATURE, 
    MAX_TOKENS, 
    SUPERVISOR_PROMPT
)

def create_supervisor_agent():
    """Create the supervisor agent to coordinate other agents - without tools"""
    
    # Initialize the model
    model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    return model  # Return the raw model instead of a React agent