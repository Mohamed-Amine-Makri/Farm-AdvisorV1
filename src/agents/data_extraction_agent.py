from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from src.config.model_config import (
    OLLAMA_BASE_URL, 
    OLLAMA_MODEL, 
    TEMPERATURE, 
    MAX_TOKENS, 
    DATA_EXTRACTION_AGENT_PROMPT
)
from langgraph.checkpoint.memory import InMemorySaver
from src.database.repository import Repository
from typing import Dict, Any, Optional
from langgraph.config import get_stream_writer

def create_data_extraction_agent():
    """Create the data extraction agent"""
    
    # Initialize the model
    model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE + 0.1,  # Slightly higher temperature for extraction
        max_tokens=MAX_TOKENS
    )
    
    # Create database tools
    def save_extracted_data(
        location: str, 
        surface_area: float, 
        soil_type: Optional[str] = None, 
        current_plants: Optional[str] = None, 
        weather_conditions: Optional[str] = None
    ) -> str:
        """Save extracted farm data to the database"""
        try:
            # Get the stream writer for providing real-time updates
            writer = get_stream_writer()
            writer(f"Processing farm data for location: {location}...")
            
            repo = Repository()
            # For simplicity, we create a new farmer if needed
            farmer = repo.create_farmer()
            
            farm = repo.save_farm_data_from_extraction(
                farmer_id=farmer.id,
                location=location,
                surface_area=float(surface_area),
                soil_type=soil_type,
                current_plants=current_plants,
                weather_conditions=weather_conditions
            )
            
            writer(f"Farm data saved successfully with ID: {farm.id}")
            repo.close()
            
            # Return farm data as a structured response
            result = {
                "farm_id": farm.id,
                "location": location,
                "surface_area": surface_area,
                "soil_type": soil_type,
                "current_plants": current_plants,
                "weather_conditions": weather_conditions
            }
            
            return f"Successfully saved farm data: {result}"
        except Exception as e:
            return f"Error saving data: {str(e)}"
    
    # Define tools
    tools = [
        Tool.from_function(
            func=save_extracted_data,
            name="save_extracted_data",
            description="Save extracted farm data to the database. Parameters: location (string), surface_area (float), soil_type (string, optional), current_plants (string, optional), weather_conditions (string, optional)"
        )
    ]
    
    # Add this function and tool
    def analyze_user_request(message: str) -> str:
        """Analyze if the user has specified what they want (recommendations, planning or both)"""
        writer = get_stream_writer()
        writer("Analyzing user request...")
        
        # Return the analysis result
        if "recommendation" in message.lower() and "plan" in message.lower():
            return "User wants both recommendations and planning"
        elif "recommendation" in message.lower():
            return "User wants recommendations only"
        elif "plan" in message.lower() or "planning" in message.lower() or "schedule" in message.lower():
            return "User wants planning only"
        else:
            return "User hasn't specified their needs clearly"

    tools.append(
        Tool.from_function(
            func=analyze_user_request,
            name="analyze_user_request",
            description="Determine if the user wants recommendations, planning, or both based on their message"
        )
    )
    
    # Create a checkpointer for short-term memory
    checkpointer = InMemorySaver()
    
    # Create the agent
    data_extraction_agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=DATA_EXTRACTION_AGENT_PROMPT,
        checkpointer=checkpointer,
    )
    
    return data_extraction_agent