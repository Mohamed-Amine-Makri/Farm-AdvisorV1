from typing import Dict, Any, TypedDict, Annotated, List, Literal, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from src.database.repository import Repository
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.config.model_config import (
    OLLAMA_MODEL, 
    OLLAMA_BASE_URL, 
    TEMPERATURE, 
    MAX_TOKENS, 
    SUPERVISOR_PROMPT,
    DATA_EXTRACTION_AGENT_PROMPT,
    PLANNING_AGENT_PROMPT,
    RECOMMENDATION_AGENT_PROMPT,
    CONVERSATIONAL_AGENT_PROMPT
)
# Import the actual agent implementations
from src.agents.Conversational_agriculture_agent import create_conversational_agent
from src.agents.data_extraction_agent import create_data_extraction_agent
from src.agents.recommendation_agent import create_recommendation_agent
from src.agents.planning_agent import create_planning_agent
from src.agents.supervisor_agent import create_supervisor_agent
from langgraph.types import Command
import uuid
import logging
import json
import time
from langgraph.store.memory import InMemoryStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a global store for long-term memory (initialize once)
memory_store = InMemoryStore()

# Define our state
class FarmAdvisorState(MessagesState):
    """State for the farm advisor system."""
    extracted_data: dict = {}
    planning_data: dict = {}
    recommendations: list = []
    active_agent: str = "supervisor"
    session_id: str = str(uuid.uuid4())
    iteration_count: int = 0  # Track iterations to prevent infinite loops

def create_supervisor_node(state: FarmAdvisorState):
    """
    Supervisor node that decides which agent to use based on the user query
    """
    # Increment iteration counter
    state_dict = dict(state)
    state_dict["iteration_count"] = state.get("iteration_count", 0) + 1
    
    # Check for too many iterations - emergency exit
    if state_dict["iteration_count"] > 10:
        logger.warning("Too many iterations detected, forcing end of processing")
        return {
            "messages": state["messages"] + [AIMessage(content="I apologize, but I'm having trouble processing your request efficiently. Could you please rephrase or provide more specific details?")],
            "active_agent": "human",
            "iteration_count": state_dict["iteration_count"]
        }
    
    # Get the most recent message
    if not state["messages"]:
        # If no messages, return to human interaction
        return {"active_agent": "human", "iteration_count": state_dict["iteration_count"]}
        
    last_message = state["messages"][-1]
    
    # Check if last message is from human
    is_human_message = False
    if hasattr(last_message, "type") and last_message.type == "human":
        is_human_message = True
    elif hasattr(last_message, "role") and last_message.role == "user":
        is_human_message = True
    
    # If it's not a user message, return to human interaction
    if not is_human_message:
        return {"active_agent": "human", "iteration_count": state_dict["iteration_count"]}
    
    # Create prompt for supervisor
    system_prompt = SUPERVISOR_PROMPT
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Based on this user query, which specialist agent should handle it? Query: {last_message.content}\n\nRespond with one of these agent names: 'conversational', 'data_extraction', 'planning', 'recommendation', or 'respond_directly'.")
    ]
    
    try:
        # Get the supervisor agent
        supervisor = create_supervisor_agent()
        response = supervisor.invoke(messages)
        
        logger.info(f"Supervisor decision: {response.content}")
        
        # Extract the agent name from the response
        agent_mapping = {
            "conversational": "conversational_agent",
            "data_extraction": "data_extraction_agent",
            "planning": "planning_agent",
            "recommendation": "recommendation_agent",
            "respond_directly": "supervisor_response"
        }
        
        # Default agent if we couldn't determine
        selected_agent = "conversational_agent"
        
        # Try to extract the agent name
        for agent_key in agent_mapping:
            if agent_key.lower() in response.content.lower():
                selected_agent = agent_mapping[agent_key]
                break
        
        logger.info(f"Selected agent: {selected_agent}")
        
        # Return the next agent to route to
        return {"active_agent": selected_agent, "iteration_count": state_dict["iteration_count"]}
    except Exception as e:
        logger.error(f"Error in supervisor: {str(e)}")
        return {
            "messages": state["messages"] + [AIMessage(content="I'm sorry, I encountered an error while processing your request. Please try again.")],
            "active_agent": "human",
            "iteration_count": state_dict["iteration_count"]
        }

def create_conversational_agent_node(state: FarmAdvisorState):
    """Handle general agricultural conversations"""
    try:
        agent = create_conversational_agent()
        
        # Create agent-specific prompt
        system_message = SystemMessage(content=CONVERSATIONAL_AGENT_PROMPT)
        messages = [system_message] + state["messages"]
        
        # Get response from the agent
        response = agent.invoke(messages)
        logger.info("Conversational agent response generated")
        
        # Add agent response to messages and reset the iteration counter
        return {
            "messages": state["messages"] + [AIMessage(content=response.content)],
            "active_agent": "human",  # Changed from END to "human" to ensure processing completes
            "iteration_count": 0
        }
    except Exception as e:
        logger.error(f"Error in conversational agent: {str(e)}")
        return {
            "messages": state["messages"] + [AIMessage(content="I'm sorry, I encountered an error while processing your request as a conversational agent.")],
            "active_agent": "human",  # Changed from END to "human"
            "iteration_count": 0
        }

def create_data_extraction_agent_node(state: FarmAdvisorState):
    """Extract structured data from user input"""
    try:
        agent = create_data_extraction_agent()
        
        # Create agent-specific prompt
        system_message = SystemMessage(content=DATA_EXTRACTION_AGENT_PROMPT)
        messages = [system_message] + state["messages"]
        
        # Get response from the agent
        response = agent.invoke(messages)
        logger.info("Data extraction agent response generated")
        
        # Try to parse extracted data as JSON
        extracted_data = {}
        try:
            # Find JSON in response
            json_start = response.content.find('{')
            json_end = response.content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response.content[json_start:json_end]
                extracted_data = json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to parse extracted data as JSON: {e}")
        
        # Generate a clean response without the JSON for the user
        user_response = "I've analyzed your information and extracted the key details."
        if extracted_data:
            user_response += " Here's what I understood from your input."
        
        return {
            "messages": state["messages"] + [AIMessage(content=user_response)],
            "extracted_data": extracted_data,
            "active_agent": "human",  # Changed from END to "human"
            "iteration_count": 0
        }
    except Exception as e:
        logger.error(f"Error in data extraction agent: {str(e)}")
        return {
            "messages": state["messages"] + [AIMessage(content="I'm sorry, I encountered an error while extracting data from your request.")],
            "active_agent": "human",  # Changed from END to "human"
            "iteration_count": 0
        }

def create_planning_agent_node(state: FarmAdvisorState):
    """Create farm planning and scheduling"""
    try:
        agent = create_planning_agent()
        
        # Combine extracted data with user message for context
        extracted_context = ""
        if state["extracted_data"]:
            extracted_context = f"\nExtracted Context: {json.dumps(state['extracted_data'])}"
        
        # Create agent-specific prompt
        system_message = SystemMessage(content=PLANNING_AGENT_PROMPT + extracted_context)
        messages = [system_message] + state["messages"]
        
        # Get response from the agent
        response = agent.invoke(messages)
        logger.info("Planning agent response generated")
        
        # Try to parse planning data if available
        planning_data = {}
        try:
            # Find JSON in response
            json_start = response.content.find('{')
            json_end = response.content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response.content[json_start:json_end]
                planning_data = json.loads(json_str)
                
                # Generate clean response for user
                user_response = response.content[:json_start].strip() + response.content[json_end:].strip()
                if not user_response:
                    user_response = "I've created a planning schedule based on your information."
            else:
                user_response = response.content
        except Exception as e:
            logger.error(f"Failed to parse planning data: {e}")
            user_response = response.content
        
        return {
            "messages": state["messages"] + [AIMessage(content=user_response)],
            "planning_data": planning_data,
            "active_agent": "human",  # Changed from END to "human"
            "iteration_count": 0
        }
    except Exception as e:
        logger.error(f"Error in planning agent: {str(e)}")
        return {
            "messages": state["messages"] + [AIMessage(content="I'm sorry, I encountered an error while creating a plan based on your request.")],
            "active_agent": "human",  # Changed from END to "human"
            "iteration_count": 0
        }

def create_recommendation_agent_node(state: FarmAdvisorState):
    """Provide specific agricultural recommendations"""
    try:
        agent = create_recommendation_agent()
        
        # Create context from extracted and planning data
        context = ""
        if state["extracted_data"]:
            context += f"Extracted Data: {json.dumps(state['extracted_data'])}\n"
        if state["planning_data"]:
            context += f"Planning Data: {json.dumps(state['planning_data'])}\n"
        
        # Create agent-specific prompt
        system_message = SystemMessage(content=RECOMMENDATION_AGENT_PROMPT + "\n" + context)
        messages = [system_message] + state["messages"]
        
        # Get response from the agent
        response = agent.invoke(messages)
        logger.info("Recommendation agent response generated")
        
        # Try to extract recommendations as a list
        recommendations = []
        try:
            # Simple heuristic to extract recommendations from the response
            lines = response.content.split("\n")
            for line in lines:
                if line.strip().startswith("- ") or line.strip().startswith("* "):
                    recommendations.append(line.strip()[2:])
        except Exception as e:
            logger.error(f"Failed to parse recommendations: {e}")
        
        return {
            "messages": state["messages"] + [AIMessage(content=response.content)],
            "recommendations": recommendations,
            "active_agent": "human",  # Changed from END to "human"
            "iteration_count": 0
        }
    except Exception as e:
        logger.error(f"Error in recommendation agent: {str(e)}")
        return {
            "messages": state["messages"] + [AIMessage(content="I'm sorry, I encountered an error while generating recommendations for your request.")],
            "active_agent": "human",  # Changed from END to "human"
            "iteration_count": 0
        }

def create_supervisor_response_node(state: FarmAdvisorState):
    """Supervisor responds directly to the user query"""
    try:
        supervisor = create_supervisor_agent()
        
        # Create prompt for direct response
        system_prompt = SUPERVISOR_PROMPT + "\nRespond directly to the user's query with helpful information."
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        
        # Get response from supervisor
        response = supervisor.invoke(messages)
        logger.info("Supervisor direct response generated")
        
        return {
            "messages": state["messages"] + [AIMessage(content=response.content)],
            "active_agent": "human",  # Changed from END to "human"
            "iteration_count": 0
        }
    except Exception as e:
        logger.error(f"Error in supervisor response: {str(e)}")
        return {
            "messages": state["messages"] + [AIMessage(content="I'm sorry, I encountered an error while processing your request directly.")],
            "active_agent": "human",  # Changed from END to "human"
            "iteration_count": 0
        }

def human_input_node(state: FarmAdvisorState):
    """Node for human interaction"""
    # Reset iteration counter when human provides input
    return {
        "active_agent": "supervisor",  # Route to supervisor for decision making
        "iteration_count": 0
    }

def route_agent(state: FarmAdvisorState):
    """Route to the next agent based on the active_agent field"""
    if state["active_agent"] == "human":
        # If human is the target, we actually want to end the graph
        return END
    return state["active_agent"]

# Create the multi-agent graph
def create_farm_advisor_graph():
    """Create the farm advisor multi-agent graph with supervisor routing"""
    
    # Initialize graph with our custom state
    builder = StateGraph(FarmAdvisorState)
    
    # Add all agent nodes
    builder.add_node("supervisor", create_supervisor_node)
    builder.add_node("conversational_agent", create_conversational_agent_node)
    builder.add_node("data_extraction_agent", create_data_extraction_agent_node)
    builder.add_node("planning_agent", create_planning_agent_node)
    builder.add_node("recommendation_agent", create_recommendation_agent_node)
    builder.add_node("supervisor_response", create_supervisor_response_node)
    builder.add_node("human", human_input_node)
    
    # Add conditional edges with END states
    builder.add_conditional_edges(
        "supervisor",
        route_agent,
        {
            "conversational_agent": "conversational_agent",
            "data_extraction_agent": "data_extraction_agent",
            "planning_agent": "planning_agent",
            "recommendation_agent": "recommendation_agent",
            "supervisor_response": "supervisor_response",
            "human": "human",  # Route to human node
            END: END,  # Allow ending the execution
        }
    )
    
    # Make sure each node has a path to END
    for node in ["conversational_agent", "data_extraction_agent", 
                 "planning_agent", "recommendation_agent", 
                 "supervisor_response"]:
        builder.add_edge(node, "human")  # Route to human instead of direct END
    
    # Start the graph with the supervisor
    builder.set_entry_point("supervisor")
    
    # Create memory to persist state between runs
    memory = InMemorySaver()
    
    return builder.compile(checkpointer=memory)

# Entry point function to run the farm advisor system
def run_farm_advisor(user_input: str = None, session_id: str = None):
    """
    Run the farm advisor system with the provided user input
    
    Args:
        user_input: The initial user input to the system
        session_id: Optional session ID for continuing a conversation
    
    Returns:
        The state after processing the input
    """
    # Set up recursion limit in config
    config = {
        "configurable": {
            "thread_id": session_id if session_id else str(uuid.uuid4()),
        },
        "recursion_limit": 50  # Higher recursion limit
    }
    
    graph = create_farm_advisor_graph()
    thread_id = config["configurable"]["thread_id"]
    
    # Initialize state
    if session_id:
        # Load existing session
        logger.info(f"Continuing session {session_id}")
        
        try:
            # First try to get previous conversation context from long-term store
            previous_messages = []
            try:
                stored_data = memory_store.get(("conversations", session_id), "messages")
                if stored_data:
                    previous_messages = stored_data.value.get("messages", [])
                    logger.info(f"Loaded {len(previous_messages)} previous messages")
            except Exception as e:
                logger.warning(f"Could not load previous messages: {e}")
            
            # Add new message if provided
            if user_input and previous_messages:
                messages = previous_messages + [HumanMessage(content=user_input)]
            elif user_input:
                messages = [HumanMessage(content=user_input)]
            else:
                messages = previous_messages
                
            state = {
                "messages": messages,
                "extracted_data": {},
                "planning_data": {},
                "recommendations": [],
                "active_agent": "supervisor",
                "session_id": session_id,
                "iteration_count": 0
            }
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            # Create new session if loading fails
            session_id = str(uuid.uuid4())
            logger.info(f"Creating new session {session_id}")
            config["configurable"]["thread_id"] = session_id
            
            # Initialize with user message if provided
            messages = []
            if user_input:
                messages = [HumanMessage(content=user_input)]
            
            state = {
                "messages": messages,
                "extracted_data": {},
                "planning_data": {},
                "recommendations": [],
                "active_agent": "supervisor",
                "session_id": session_id,
                "iteration_count": 0
            }
    else:
        # New session
        session_id = str(uuid.uuid4())
        logger.info(f"Starting new session {session_id}")
        config["configurable"]["thread_id"] = session_id
        
        # Initialize with user message if provided
        messages = []
        if user_input:
            messages = [HumanMessage(content=user_input)]
        
        state = {
            "messages": messages,
            "extracted_data": {},
            "planning_data": {},
            "recommendations": [],
            "active_agent": "supervisor" if user_input else "human",
            "session_id": session_id,
            "iteration_count": 0
        }
    
    # Throttle to prevent Ollama overload
    time.sleep(0.5)
    
    # Run the graph
    try:
        result = graph.invoke(state, config)
        # Make sure the session ID is returned
        result["session_id"] = session_id
        
        # Store conversation in long-term memory
        try:
            memory_store.put(
                ("conversations", session_id),
                "messages",
                {"messages": result.get("messages", [])}
            )
            # Also store any extracted data for future reference
            if result.get("extracted_data"):
                memory_store.put(
                    ("user_data", session_id),
                    "extracted_data",
                    result["extracted_data"]
                )
        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")
            
        return result
    except Exception as e:
        logger.error(f"Error running graph: {e}")
        # Return basic state with error message
        return {
            "messages": state["messages"] + [AIMessage(content="I'm sorry, I encountered an error. Please try again.")],
            "session_id": session_id
        }


