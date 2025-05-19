from langchain_ollama import ChatOllama
from src.config.model_config import (
    OLLAMA_BASE_URL, 
    OLLAMA_MODEL, 
    TEMPERATURE, 
    MAX_TOKENS,
    RECOMMENDATION_AGENT_PROMPT
)
from langgraph.checkpoint.memory import InMemorySaver

def create_recommendation_agent():
    """Create a recommendation agent using only the LLM and the centralized prompt."""

    # Initialize the model
    model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    # Return the model directly, no tools
    return model