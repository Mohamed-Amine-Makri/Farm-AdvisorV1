from typing import Annotated, Dict, List, Any, Optional, Union
from langchain_core.messages import AnyMessage, BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from src.utils.langgraph_utils import create_handoff_tool
from langgraph.types import Command
from src.agents.supervisor_agent import create_supervisor_agent
from src.agents.Conversational_agriculture_agent import create_conversational_agent
from src.agents.data_extraction_agent import create_data_extraction_agent
from src.agents.recommendation_agent import create_recommendation_agent
from src.agents.planning_agent import create_planning_agent
from src.database.repository import Repository
from langchain_ollama import ChatOllama
from src.config.model_config import (
    OLLAMA_MODEL, 
    OLLAMA_BASE_URL, 
    TEMPERATURE, 
    MAX_TOKENS, 
    CONVERSATIONAL_AGENT_PROMPT,
    DATA_EXTRACTION_AGENT_PROMPT,
    RECOMMENDATION_AGENT_PROMPT,
    PLANNING_AGENT_PROMPT,
    COMBINED_PROMPT
)
from langgraph.store.memory import InMemoryStore
from src.utils.ollama_utils import check_model_tool_support
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_multi_agent_graph():
    """Create a multi-agent system using LangGraph"""
    
    try:
        # Check if the model supports tools
        has_tool_support = check_model_tool_support()
        
        if has_tool_support:
            logger.info(f"Model {OLLAMA_MODEL} supports tools. Creating full agent graph...")
            return create_full_agent_graph()
        else:
            logger.info(f"Model {OLLAMA_MODEL} does not support tools. Creating simple graph...")
            return create_simple_graph()
    except Exception as e:
        logger.error(f"Error checking tool support: {str(e)}. Falling back to simple graph.")
        return create_simple_graph()

def create_full_agent_graph():
    """Create the full multi-agent graph with tool support"""
    
    # Define state that includes messages and thread_id for persistence
    class FarmAdvisorState(MessagesState):
        thread_id: str = None
        farm_data_extracted: bool = False
        user_needs_specified: bool = False
        last_active_agent: str = "conversational_agent"
        
    # Create a checkpointer for the entire graph
    checkpointer = InMemorySaver()
    
    # Create a store for long-term memory
    memory_store = InMemoryStore()
    
    # Define tools for each agent to pass control to another agent
    transfer_to_conversational = create_handoff_tool(
        agent_name="conversational_agent", 
        description="Transfer to conversational agent for greetings and general agriculture questions"
    )
    
    transfer_to_data_extraction = create_handoff_tool(
        agent_name="data_extraction_agent", 
        description="Transfer to data extraction agent to extract farm details"
    )
    
    transfer_to_recommendation = create_handoff_tool(
        agent_name="recommendation_agent", 
        description="Transfer to recommendation agent for plant and irrigation recommendations"
    )
    
    transfer_to_planning = create_handoff_tool(
        agent_name="planning_agent", 
        description="Transfer to planning agent for creating agricultural plans"
    )
    
    try:
        # Create all the specialized agents
        data_extraction_agent = create_data_extraction_agent()
        recommendation_agent = create_recommendation_agent()
        planning_agent = create_planning_agent()
        conversational_agent = create_conversational_agent()
        
        # Create the supervisor agent
        supervisor_tools = [
            {
                "name": "route_to_conversational_agent",
                "description": "Route to conversational agent for greetings, farewells, and general agricultural questions"
            },
            {
                "name": "route_to_data_extraction_agent",
                "description": "Route to data extraction agent to extract farm information from user messages"
            },
            {
                "name": "route_to_recommendation_agent",
                "description": "Route to recommendation agent for plant and irrigation recommendations"
            },
            {
                "name": "route_to_planning_agent",
                "description": "Route to planning agent for creating agricultural planning schedules"
            }
        ]
        
        supervisor_agent = create_supervisor_agent(supervisor_tools)
        
        # Function to save conversation to database
        def save_conversation(state: Annotated[FarmAdvisorState, "state"]) -> Dict:
            """Save conversation history to the database"""
            try:
                messages = state["messages"]
                # Get the last user message and last assistant message
                user_messages = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "user"]
                assistant_messages = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "assistant"]
                
                if user_messages and assistant_messages:
                    last_user_message = user_messages[-1].get("content", "")
                    last_assistant_message = assistant_messages[-1].get("content", "")
                    
                    repo = Repository()
                    # Use existing thread ID if available in state or create a new thread
                    thread_id = state.get("thread_id", str(uuid.uuid4()))
                    
                    conversation = repo.save_conversation_history(
                        thread_id=thread_id,
                        user_message=last_user_message,
                        assistant_message=last_assistant_message
                    )
                    
                    repo.close()
                    return {"thread_id": conversation.thread_id}
                return {}
            except Exception as e:
                logger.error(f"Error saving conversation: {str(e)}")
                return {}
        
        # Create the graph
        workflow = StateGraph(FarmAdvisorState)
        
        # Add all nodes
        workflow.add_node("supervisor", supervisor_agent)
        workflow.add_node("conversational_agent", conversational_agent)
        workflow.add_node("data_extraction_agent", data_extraction_agent)
        workflow.add_node("recommendation_agent", recommendation_agent)
        workflow.add_node("planning_agent", planning_agent)
        workflow.add_node("save_conversation", save_conversation)
        
        # Route to appropriate agent based on supervisor's decision
        def route_to_agent(state: Annotated[FarmAdvisorState, "state"]) -> str:
            """Route to appropriate agent based on supervisor's decision"""
            try:
                # Check if we've already made multiple transitions, if so, end to avoid infinite loops
                if len(state.get("_loops", [])) > 10:
                    logger.warning("Detected potential loop in agent routing, forcing conversational agent")
                    return "conversational_agent"
            
                # Invoke the supervisor to get routing decision
                result = supervisor_agent.invoke({"messages": state["messages"]})
                
                # Check if farm data was recently extracted but user needs weren't specified
                if state.get("farm_data_extracted") and not state.get("user_needs_specified"):
                    # Instead of asking what to do with the information, proactively route to recommendation agent
                    state["user_needs_specified"] = True
                    return "recommendation_agent"
                
                # Handle different response types from supervisor
                if isinstance(result, dict) and "messages" in result:
                    for message in result["messages"]:
                        if isinstance(message, dict) and message.get("role") == "assistant" and message.get("tool_calls"):
                            for tool_call in message.get("tool_calls", []):
                                function_name = tool_call.get("function", {}).get("name")
                                if function_name:
                                    if function_name == "route_to_data_extraction_agent":
                                        return "data_extraction_agent"
                                    elif function_name == "route_to_conversational_agent":
                                        return "conversational_agent"
                                    elif function_name == "route_to_recommendation_agent":
                                        return "recommendation_agent"
                                    elif function_name == "route_to_planning_agent":
                                        return "planning_agent"
                elif isinstance(result, AIMessage):
                    # Handle AIMessage type properly
                    if hasattr(result, 'additional_kwargs') and 'tool_calls' in result.additional_kwargs:
                        tool_calls = result.additional_kwargs.get('tool_calls', [])
                        for tool_call in tool_calls:
                            function_name = tool_call.get("function", {}).get("name")
                            if function_name:
                                if function_name == "route_to_data_extraction_agent":
                                    return "data_extraction_agent"
                                elif function_name == "route_to_conversational_agent":
                                    return "conversational_agent"
                                elif function_name == "route_to_recommendation_agent":
                                    return "recommendation_agent"
                                elif function_name == "route_to_planning_agent":
                                    return "planning_agent"
                        
            except Exception as e:
                logger.error(f"Error in route_to_agent: {str(e)}")
            
            # Default to conversational agent
            return "conversational_agent"
        
        # Define the edges in the graph
        workflow.add_edge(START, "supervisor")
        
        # Connect supervisor to agents based on routing decisions
        workflow.add_conditional_edges(
            "supervisor",
            route_to_agent,
            {
                "conversational_agent": "conversational_agent",
                "data_extraction_agent": "data_extraction_agent", 
                "recommendation_agent": "recommendation_agent",
                "planning_agent": "planning_agent",
                "save_conversation": "save_conversation"
            }
        )
        
        # Connect all agents to END to provide a definitive stopping point
        workflow.add_edge("conversational_agent", END)
        workflow.add_edge("data_extraction_agent", "save_conversation")
        workflow.add_edge("recommendation_agent", "save_conversation")
        workflow.add_edge("planning_agent", "save_conversation")
        
        # Connect save_conversation to END instead of back to supervisor
        workflow.add_edge("save_conversation", END)
        
        # Compile the graph without recursion_limit parameter (not supported in this version)
        logger.info("Compiling full agent graph...")
        multi_agent_graph = workflow.compile(
            checkpointer=checkpointer
        )
        return multi_agent_graph
    except Exception as e:
        logger.error(f"Error creating full agent graph: {str(e)}")
        logger.info("Falling back to simple graph...")
        return create_simple_graph()

def create_simple_graph():
    """Create a simple graph that works without tool calling"""
    class FarmAdvisorState(MessagesState):
        thread_id: str = None
    
    try:
        # Create a checkpointer
        checkpointer = InMemorySaver()
        
        # Create a simple conversational model
        model = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        # Simple conversation function that handles all requests
        def simple_agent(state):
            try:
                messages = state.get("messages", [])
                
                # Add the combined prompt as a system message for the first message
                if len([m for m in messages if isinstance(m, dict) and m.get("role") == "user"]) <= 1:
                    # Prepend system message only once at the start of conversation
                    system_message = {"role": "system", "content": COMBINED_PROMPT}
                    if not any(m.get("role") == "system" for m in messages if isinstance(m, dict)):
                        messages = [system_message] + messages
                
                # Use the model directly since we can't use tools
                logger.info(f"Simple agent processing message with model {OLLAMA_MODEL}...")
                response = model.invoke(messages)
                
                # Handle different response types safely
                new_message = None
                
                if isinstance(response, str):
                    # String response
                    logger.info("Got string response")
                    new_message = {"role": "assistant", "content": response}
                elif isinstance(response, AIMessage):
                    # AIMessage response 
                    logger.info("Got AIMessage response")
                    new_message = {"role": "assistant", "content": response.content if hasattr(response, 'content') else str(response)}
                elif isinstance(response, BaseMessage):
                    # Other BaseMessage response
                    logger.info("Got BaseMessage response")
                    new_message = {"role": "assistant", "content": str(response)}
                elif isinstance(response, dict) and "content" in response:
                    # Dictionary with content
                    logger.info("Got dict response")
                    new_message = {"role": "assistant", "content": response["content"]}
                else:
                    # Fallback for any other type
                    logger.info(f"Got unrecognized response type: {type(response)}")
                    new_message = {"role": "assistant", "content": f"Response received: {str(response)}"}
                
                logger.info(f"Final message: {new_message}")
                
                # Save conversation to database
                try:
                    repo = Repository()
                    # Use existing thread ID if available in state or create a new thread
                    thread_id = state.get("thread_id", str(uuid.uuid4()))
                    
                    # Extract recent messages
                    user_messages = [m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "user"]
                    if user_messages:
                        last_user_message = user_messages[-1]
                        assistant_message = new_message.get("content", "")
                        
                        conversation = repo.save_conversation_history(
                            thread_id=thread_id,
                            user_message=last_user_message,
                            assistant_message=assistant_message
                        )
                        
                        thread_id = conversation.thread_id
                        
                    repo.close()
                except Exception as e:
                    logger.error(f"Error saving conversation in simple graph: {str(e)}")
                
                # Return new message and thread_id
                return {
                    "messages": [new_message],
                    "thread_id": thread_id
                }
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                logger.error(error_message)
                return {"messages": [{"role": "assistant", "content": f"I apologize for the error: {error_message}. Please try again."}]}
        
        # Create the graph
        workflow = StateGraph(FarmAdvisorState)
        workflow.add_node("conversation", simple_agent)
        workflow.add_edge(START, "conversation")
        workflow.add_edge("conversation", END)
        
        # Compile the graph without recursion_limit parameter (not supported in this version)
        logger.info("Compiling simple graph...")
        return workflow.compile(
            checkpointer=checkpointer
        )
    except Exception as e:
        logger.error(f"Error creating simple graph: {str(e)}")
        raise

# Define an alias for backward compatibility
MultiAgentGraph = create_multi_agent_graph