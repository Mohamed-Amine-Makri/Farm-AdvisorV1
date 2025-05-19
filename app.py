import os
import time
import sys
import uuid
import json
from colorama import Fore, Style, init
from src.graph.multi_agent_graph import run_farm_advisor
from src.database.models import init_db
from src.config.model_config import OLLAMA_BASE_URL, OLLAMA_MODEL
from src.utils.ollama_utils import check_ollama_availability
from langchain_core.messages import AIMessage, HumanMessage
import logging
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize colorama for cross-platform colored terminal output
init()

# After other logging setup, redirect log output to a file
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, "farm_advisor.log"))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)
logging.getLogger().setLevel(logging.INFO)

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

def show_thinking_animation(stop_event):
    """Display a "thinking" animation while the model is processing"""
    dots = 0
    while not stop_event.is_set():
        dots = (dots % 3) + 1
        print_colored(f"\rThinking{'.' * dots}{' ' * (3 - dots)}", Fore.YELLOW, end="")
        time.sleep(0.5)
    # Clear the line when done
    print_colored("\r" + " " * 20 + "\r", end="")

def extract_ai_message(messages):
    """Extract the last AI message from the messages list"""
    if not messages:
        return "No response generated."
        
    # Debug message types
    logger.debug(f"Message types: {[type(m).__name__ for m in messages]}")
    
    # Try different extraction methods
    for msg in reversed(messages):
        # Handle LangChain AIMessage
        if isinstance(msg, AIMessage):
            return msg.content
            
        # Check for type attribute (LangChain structure)
        if hasattr(msg, "type") and msg.type == "ai":
            return msg.content
            
        # Check for role attribute (OpenAI format)
        if hasattr(msg, "role") and msg.role == "assistant":
            return msg.content
            
        # Check dictionary format (various APIs)
        if isinstance(msg, dict):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
            if msg.get("type") == "ai":
                return msg.get("content", "")
    
    # If no AI message is found, return a default message with recommendations if available
    if any(msg for msg in messages if hasattr(msg, "content")):
        # Get the last message with content as fallback
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                return msg.content
                
    return "No response generated."

def main():
    """Main entry point for the Farm Advisor application"""
    print_banner()
    
    # Check if Ollama is available
    print_colored("Checking Ollama availability...", Fore.YELLOW)
    if not check_ollama_availability():
        print_colored(f"Error: Cannot connect to Ollama at {OLLAMA_BASE_URL} or model {OLLAMA_MODEL} is not available.", Fore.RED)
        print_colored("Please ensure Ollama is running and the model is installed.", Fore.RED)
        return 1
    
    print_colored(f"✓ Connected to Ollama with model {OLLAMA_MODEL}", Fore.GREEN)
    
    # Initialize database if needed
    try:
        init_db()
        print_colored("✓ Database initialized", Fore.GREEN)
    except Exception as e:
        logger.warning(f"Database initialization failed: {str(e)}")
        print_colored(f"Note: Database initialization skipped - {str(e)}", Fore.YELLOW)
    
    print_colored("\nFarm Advisor is ready! How can I help with your farming needs today?", Fore.CYAN)
    
    # Session management
    session_id = None
    
    # Main conversation loop
    while True:
        try:
            # Get user input
            print_colored("\nYou: ", Fore.GREEN, end="")
            user_input = input().strip()
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                print_colored("\nThank you for using Farm Advisor. Goodbye!", Fore.CYAN)
                break
            
            if not user_input:  # Skip empty inputs
                continue
                
            # Create and start the thinking animation
            stop_animation = threading.Event()
            animation_thread = threading.Thread(target=show_thinking_animation, args=(stop_animation,))
            animation_thread.daemon = True
            animation_thread.start()
            
            try:
                # Process the input through our multi-agent system
                start_time = time.time()
                result = run_farm_advisor(user_input, session_id)
                
                # Save the session ID for next interaction
                session_id = result.get("session_id", session_id)
                
                # Calculate response time
                response_time = time.time() - start_time
                
            except Exception as e:
                # Stop the animation before raising the exception
                stop_animation.set()
                animation_thread.join(timeout=1.0)
                logger.error(f"Error processing input: {str(e)}", exc_info=True)
                print_colored(f"\nSorry, an error occurred: {str(e)}", Fore.RED)
                continue
            finally:
                # Ensure animation stops
                stop_animation.set()
                animation_thread.join(timeout=1.0)
            
            # Extract and display the response
            ai_response = extract_ai_message(result.get("messages", []))
            
            print_colored("\nFarm Advisor: ", Fore.BLUE)
            print_colored(ai_response, Fore.WHITE)
            print_colored(f"\n(Response time: {response_time:.2f}s)", Fore.CYAN)
            
            # Display recommendations if available
            # if result.get("recommendations") and len(result.get("recommendations", [])) > 0:
            #     print_colored("\nKey Recommendations:", Fore.MAGENTA)
            #     for i, rec in enumerate(result["recommendations"][:5], 1):  # Show top 5 recommendations
            #         print_colored(f"  {i}. {rec}", Fore.CYAN)
            
            # If in debug mode, show extracted or planning data
            if os.environ.get("DEBUG_MODE") == "1":
                if result.get("extracted_data"):
                    print_colored("\n[DEBUG] Extracted Data:", Fore.YELLOW)
                    print_colored(json.dumps(result["extracted_data"], indent=2), Fore.WHITE)
                
                if result.get("planning_data"):
                    print_colored("\n[DEBUG] Planning Data:", Fore.YELLOW)
                    print_colored(json.dumps(result["planning_data"], indent=2), Fore.WHITE)
                    
                print_colored(f"\n[DEBUG] Active Agent: {result.get('active_agent', 'unknown')}", Fore.YELLOW)

        except KeyboardInterrupt:
            print_colored("\nExiting Farm Advisor...", Fore.YELLOW)
            break
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            print_colored(f"\nAn unexpected error occurred: {str(e)}", Fore.RED)
            print_colored("The application will continue running.", Fore.YELLOW)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_colored("\nApplication terminated by user.", Fore.YELLOW)
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        print_colored(f"\nA fatal error occurred: {str(e)}", Fore.RED)
        sys.exit(1)
