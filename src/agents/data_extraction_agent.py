from langchain_ollama import ChatOllama
from src.config.model_config import (
    OLLAMA_BASE_URL, 
    OLLAMA_MODEL, 
    TEMPERATURE, 
    MAX_TOKENS, 
    DATA_EXTRACTION_AGENT_PROMPT
)
from langgraph.checkpoint.memory import InMemorySaver

def create_data_extraction_agent():
    """Create the data extraction agent without tools"""
    
    # Initialize the model
    model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE + 0.1,  # Slightly higher temperature for extraction
        max_tokens=MAX_TOKENS
    )
    
    # Create the agent without tools
    checkpointer = InMemorySaver()
    
    # Instead of creating a react agent with tools, use just the model
    return model