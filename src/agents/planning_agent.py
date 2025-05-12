from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from src.config.model_config import (
    OLLAMA_BASE_URL, 
    OLLAMA_MODEL, 
    TEMPERATURE, 
    MAX_TOKENS, 
    PLANNING_AGENT_PROMPT
)
from langgraph.checkpoint.memory import InMemorySaver
from src.database.repository import Repository
from typing import Dict, Any, Optional
from langgraph.config import get_stream_writer

def create_planning_agent():
    """Create the planning agent"""
    
    # Initialize the model
    model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    # Create database tools
    def save_plan(
        farm_id: int,
        monthly_schedule: str,
        planting_schedule: Optional[str] = None,
        irrigation_schedule: Optional[str] = None,
        soil_preparation: Optional[str] = None,
        harvest_times: Optional[str] = None,
        seasonal_considerations: Optional[str] = None
    ) -> str:
        """Save agricultural plan to the database"""
        try:
            # Get the stream writer for providing real-time updates
            writer = get_stream_writer()
            writer(f"Processing agricultural plan for farm ID: {farm_id}...")
            
            repo = Repository()
            plan = repo.create_plan(
                farm_id=farm_id,
                monthly_schedule=monthly_schedule,
                planting_schedule=planting_schedule,
                irrigation_schedule=irrigation_schedule,
                soil_preparation=soil_preparation,
                harvest_times=harvest_times,
                seasonal_considerations=seasonal_considerations
            )
            
            writer(f"Plan saved successfully with ID: {plan.id}")
            repo.close()
            
            return f"Successfully saved agricultural plan with ID: {plan.id}"
        except Exception as e:
            return f"Error saving plan: {str(e)}"
    
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
            func=save_plan,
            name="save_plan",
            description="Save agricultural plan to the database. Parameters: farm_id (int), monthly_schedule (string), planting_schedule (string, optional), irrigation_schedule (string, optional), soil_preparation (string, optional), harvest_times (string, optional), seasonal_considerations (string, optional)"
        ),
        Tool.from_function(
            func=get_latest_farm_data,
            name="get_latest_farm_data",
            description="Get the most recent farm data from the database to create a planning schedule"
        )
    ]
    
    # Create a checkpointer for short-term memory
    checkpointer = InMemorySaver()
    
    # Create the agent
    planning_agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=PLANNING_AGENT_PROMPT,
        checkpointer=checkpointer,
    )
    
    return planning_agent