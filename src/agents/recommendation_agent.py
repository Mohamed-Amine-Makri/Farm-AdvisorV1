from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from src.config.model_config import (
    OLLAMA_BASE_URL, 
    OLLAMA_MODEL, 
    TEMPERATURE, 
    MAX_TOKENS, 
    RECOMMENDATION_AGENT_PROMPT
)
from langgraph.checkpoint.memory import InMemorySaver
from src.database.repository import Repository
from typing import Dict, Any, Optional
from langgraph.config import get_stream_writer

def create_recommendation_agent():
    """Create the recommendation agent"""
    
    # Initialize the model
    model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    # Create database tools
    def save_recommendation(
        farm_id: int,
        recommended_plants: str,
        irrigation_methods: str,
        reasoning: Optional[str] = None
    ) -> str:
        """Save recommendation to the database"""
        try:
            # Get the stream writer for providing real-time updates
            writer = get_stream_writer()
            writer(f"Processing recommendation for farm ID: {farm_id}...")
            
            repo = Repository()
            recommendation = repo.create_recommendation(
                farm_id=farm_id,
                recommended_plants=recommended_plants,
                irrigation_methods=irrigation_methods,
                reasoning=reasoning
            )
            
            writer(f"Recommendation saved successfully with ID: {recommendation.id}")
            repo.close()
            
            return f"Successfully saved recommendation with ID: {recommendation.id}"
        except Exception as e:
            return f"Error saving recommendation: {str(e)}"
    
    def get_latest_farm_data() -> str:
        """Get the most recent farm data from the database"""
        try:
            repo = Repository()
            farm = repo.get_latest_farm()
            if not farm:
                repo.close()
                return "No farm data available. Please provide farm details first."
            
            farm_data = {
                "farm_id": farm.id,
                "location": farm.location,
                "surface_area": farm.surface_area,
                "soil_type": farm.soil_type,
                "current_plants": farm.current_plants,
                "weather_conditions": farm.weather_conditions
            }
            
            repo.close()
            return str(farm_data)
        except Exception as e:
            return f"Error retrieving farm data: {str(e)}"
    
    # Define tools
    tools = [
        Tool.from_function(
            func=save_recommendation,
            name="save_recommendation",
            description="Save plant and irrigation recommendations to the database. Parameters: farm_id (int), recommended_plants (string), irrigation_methods (string), reasoning (string, optional)"
        ),
        Tool.from_function(
            func=get_latest_farm_data,
            name="get_latest_farm_data",
            description="Get the most recent farm data from the database to make recommendations"
        )
    ]
    
    # Create a checkpointer for short-term memory
    checkpointer = InMemorySaver()
    
    # Create the agent
    recommendation_agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=RECOMMENDATION_AGENT_PROMPT,
        checkpointer=checkpointer,
    )
    
    return recommendation_agent