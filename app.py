import os
import time
import sys
import uuid
from colorama import Fore, Style, init
from src.graph.multi_agent_graph import create_multi_agent_graph, create_simple_graph
from src.database.models import init_db
from src.config.model_config import OLLAMA_BASE_URL, OLLAMA_MODEL
from src.utils.ollama_utils import check_ollama_availability
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize colorama for cross-platform colored terminal output
init()

def print_colored(text, color=Fore.WHITE, end="\n"):
    """Print text with specified color"""
    print(f"{color}{text}{Style.RESET_ALL}", end=end)
    sys.stdout.flush()  # Ensure output appears immediately

def print_banner():
    """Print welcome banner"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print_colored("""
    ╔════════════════════════════════════════════════╗
    ║                                                ║
    ║            TUNISIAN FARM ADVISOR               ║
    ║       Agricultural Planning & Recommendations  ║
    ║                                                ║
    ╚════════════════════════════════════════════════╝
    """, Fore.GREEN)
    print_colored("Welcome! Ask me anything about farming in Tunisia or describe your farm for personalized recommendations and planning.", Fore.CYAN)
    print_colored("Type 'exit' to quit.\n", Fore.YELLOW)

def direct_chat_mode():
    """
    Fallback mode that directly uses Ollama API without the graph
    when other methods aren't available
    """
    from src.utils.ollama_utils import get_ollama_chat_completion
    from src.config.model_config import COMBINED_PROMPT
    
    print_colored("Running in direct chat mode (minimal functionality)...", Fore.YELLOW)
    messages = [{"role": "system", "content": COMBINED_PROMPT}]
    
    # Add initial greeting to conversation
    assistant_greeting = "Hello! I'm your Tunisian Farm Advisor. I can help with agricultural questions, provide crop recommendations, or create farming plans. How can I assist you today?"
    print_colored("Assistant: ", Fore.BLUE, end="")
    print_colored(assistant_greeting, Fore.BLUE)
    print()
    
    messages.append({"role": "assistant", "content": assistant_greeting})
    
    # Main interaction loop
    while True:
        # Get user input
        user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            print_colored("Thank you for using the Tunisian Farm Advisor. Goodbye!", Fore.CYAN)
            break
        
        # Add user message to conversation history
        messages.append({"role": "user", "content": user_input})
        
        # Prepare to show assistant's response
        print_colored("Assistant: ", Fore.BLUE, end="")
        
        try:
            # Get direct chat completion
            response = get_ollama_chat_completion(messages)
            
            # Print response and add to messages
            print_colored(response, Fore.BLUE)
            messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            logger.exception("Error in direct chat mode")
            error_msg = f"I'm sorry, I encountered an error: {str(e)}"
            print_colored(error_msg, Fore.RED)
            # Don't add error to message history
        
        print()  # Extra line for readability

def main():
    """Main application entry point"""
    print_banner()
    
    # Initialize the database
    print_colored("Initializing database...", Fore.CYAN)
    init_db()
    
    # Check Ollama availability
    print_colored("Checking Ollama availability...", Fore.CYAN)
    ollama_ready = check_ollama_availability()
    
    if not ollama_ready:
        print_colored("Warning: Ollama is not responding correctly. Trying direct mode...", Fore.YELLOW)
        direct_chat_mode()
        return
    
    # Create the agent system
    print_colored("Initializing AI Farm Advisor...", Fore.CYAN)
    try:
        agent_system = create_multi_agent_graph()
    except Exception as e:
        print_colored(f"Error initializing agent system: {str(e)}", Fore.RED)
        print_colored("Falling back to direct chat mode...", Fore.YELLOW)
        direct_chat_mode()
        return
    
    # Initialize conversation with UUID
    thread_id = str(uuid.uuid4())
    messages = []
    
    # Print initial greeting
    print_colored("Assistant: ", Fore.BLUE, end="")
    print_colored("Hello! I'm your Tunisian Farm Advisor. I can help with agricultural questions, provide crop recommendations, or create farming plans. How can I assist you today?", Fore.BLUE)
    print()
    
    # Add initial greeting to conversation
    messages.append({
        "role": "assistant",
        "content": "Hello! I'm your Tunisian Farm Advisor. I can help with agricultural questions, provide crop recommendations, or create farming plans. How can I assist you today?"
    })
    
    # Main interaction loop
    while True:
        # Get user input
        user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            print_colored("Thank you for using the Tunisian Farm Advisor. Goodbye!", Fore.CYAN)
            break
        
        # Add user message to conversation history
        messages.append({"role": "user", "content": user_input})
        
        # Prepare to show assistant's response
        print_colored("Assistant: ", Fore.BLUE, end="")
        
        # Configure the agent system with thread_id and checkpoint_id
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": thread_id
            }
        }
        
        # Track response content
        collecting_response = ""
        
        try:
            # Stream the agent system's response with more robust error handling
            for chunk in agent_system.stream(
                {"messages": messages},
                config=config,
                stream_mode="updates"
            ):
                # Handle different types of chunks
                if isinstance(chunk, str):
                    # If plain string, print directly
                    print_colored(chunk, Fore.BLUE, end="")
                    collecting_response += chunk
                    
                elif isinstance(chunk, dict):
                    # For updates and messages
                    if "messages" in chunk:
                        all_msgs = chunk.get("messages", [])
                        assistant_msgs = []
                        
                        # Handle various message formats
                        for msg in all_msgs:
                            if isinstance(msg, dict) and msg.get("role") == "assistant" and "content" in msg:
                                assistant_msgs.append(msg)
                            elif hasattr(msg, "type") and msg.type == "assistant" and hasattr(msg, "content"):
                                assistant_msgs.append({"role": "assistant", "content": msg.content})
                            elif hasattr(msg, "role") and msg.role == "assistant" and hasattr(msg, "content"):
                                assistant_msgs.append({"role": "assistant", "content": msg.content})
                        
                        if assistant_msgs:
                            latest_msg = assistant_msgs[-1]
                            content = latest_msg.get("content", "")
                            if content and not collecting_response:
                                print_colored(content, Fore.BLUE, end="")
                                collecting_response = content
                    
                    # Extract thread_id for persistence
                    if "thread_id" in chunk and chunk["thread_id"]:
                        thread_id = chunk["thread_id"]
                    
                    # Extract response content from various possible formats
                    if not collecting_response:
                        if "content" in chunk:
                            print_colored(chunk["content"], Fore.BLUE, end="")
                            collecting_response = chunk["content"]
                        elif "response" in chunk:
                            print_colored(chunk["response"], Fore.BLUE, end="")
                            collecting_response = chunk["response"]
                
                time.sleep(0.01)  # Small delay for more natural output pacing
            
            print()  # End the line after streaming completes
            
            # Add the final response to messages if we got meaningful content
            if collecting_response:
                # Check if this content is already in messages
                if not any(
                    (isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content") == collecting_response)
                    for msg in messages
                ):
                    messages.append({
                        "role": "assistant", 
                        "content": collecting_response
                    })
            
        except Exception as e:
            logger.exception("Error processing response")
            print_colored(f"\nError: {str(e)}", Fore.RED)
            # Add error message to conversation
            error_msg = f"I'm sorry, I encountered an error: {str(e)}"
            print_colored(error_msg, Fore.BLUE)
            messages.append({"role": "assistant", "content": error_msg})
            
            # If we get timeout errors repeatedly, fall back to direct chat mode
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                retry = input(f"{Fore.YELLOW}Connection issues detected. Switch to direct chat mode? (y/n): {Style.RESET_ALL}")
                if retry.lower() in ["y", "yes"]:
                    direct_chat_mode()
                    return
        
        print()  # Extra line for readability between interactions

if __name__ == "__main__":
    main()