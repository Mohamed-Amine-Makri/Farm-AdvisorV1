from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from src.config.model_config import (
    OLLAMA_BASE_URL, 
    OLLAMA_MODEL, 
    TEMPERATURE, 
    MAX_TOKENS, 
    SUPERVISOR_PROMPT
)
from typing import Dict, List, Any

def create_supervisor_agent(tools: List[Dict[str, Any]]):
    """Create the supervisor agent to coordinate other agents"""
    
    # Initialize the model
    model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    # Create the agent
    supervisor_agent = create_react_agent(
        model=model,
        tools=[
            Tool.from_function(
                func=lambda target_node: {"target": target_node},
                name=tool["name"],
                description=tool["description"]
            )
            for tool in tools
        ],
        prompt=SUPERVISOR_PROMPT,
    )
    
    return supervisor_agent