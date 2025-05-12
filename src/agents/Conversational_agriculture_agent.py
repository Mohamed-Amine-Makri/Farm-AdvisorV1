from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from src.config.model_config import (
    OLLAMA_BASE_URL, 
    OLLAMA_MODEL, 
    TEMPERATURE, 
    MAX_TOKENS, 
    CONVERSATIONAL_AGENT_PROMPT
)

def create_conversational_agent():
    """Create the conversational agriculture agent"""
    
    # Initialize the model
    model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    # Create a checkpointer for short-term memory
    checkpointer = InMemorySaver()
    
    # Create the agent with an empty tools list (will be populated in multi_agent_graph.py)
    conversational_agent = create_react_agent(
        model=model,
        tools=[],  # Add empty tools list here
        prompt=CONVERSATIONAL_AGENT_PROMPT,
        checkpointer=checkpointer,
    )
    
    return conversational_agent