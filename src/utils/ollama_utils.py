import requests
import json
import time
import logging
import httpx
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

from src.config.model_config import (
    OLLAMA_BASE_URL, 
    OLLAMA_MODEL,
    MAX_TOKENS
)

def check_ollama_availability() -> bool:
    """
    Check if Ollama server is available and the desired model is loaded
    
    Returns:
        bool: True if Ollama is ready, False otherwise
    """
    try:
        logger.info(f"Ensuring Ollama is ready (timeout=20s)...")
        
        # Step 1: Check if the server is running (give it 3 tries)
        max_attempts = 3
        attempt = 1
        server_ready = False
        
        while attempt <= max_attempts and not server_ready:
            try:
                response = requests.get(
                    f"{OLLAMA_BASE_URL}/api/tags",
                    timeout=20
                )
                if response.status_code == 200:
                    logger.info(f"Ollama server check: {response.status_code}")
                    server_ready = True
                else:
                    logger.warning(f"Ollama server returned status code: {response.status_code}")
                    time.sleep(3)  # Wait 3 seconds before retrying
            except Exception as e:
                logger.warning(f"Attempt {attempt}: Ollama server check failed: {str(e)}")
                time.sleep(3)  # Wait 3 seconds before retrying
            attempt += 1
        
        if not server_ready:
            logger.error(f"Ollama server is not available after {max_attempts} attempts")
            return False
        
        # Step 2: Find if our model is available
        model_available = False
        try:
            response = requests.get(
                f"{OLLAMA_BASE_URL}/api/tags",
                timeout=20
            )
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                
                # Check if the exact model name is available
                if OLLAMA_MODEL in model_names:
                    model_available = True
                    logger.info(f"Found matching model: {OLLAMA_MODEL}")
                else:
                    # Try to find a similar model if exact match not found
                    # Many systems might have llama3:latest instead of llama3.1:8b
                    base_name = OLLAMA_MODEL.split(":")[0].split(".")[0]
                    matching_models = [model for model in model_names if base_name in model]
                    
                    if matching_models:
                        alternative_model = matching_models[0]
                        logger.warning(f"Exact model {OLLAMA_MODEL} not found, using alternative: {alternative_model}")
                        model_available = True
                    else:
                        logger.error(f"Model {OLLAMA_MODEL} not found and no alternatives available")
            else:
                logger.error(f"Failed to get model list: {response.status_code}")
        except Exception as e:
            logger.error(f"Error checking model availability: {str(e)}")
            return False
            
        if not model_available:
            logger.error(f"Required model '{OLLAMA_MODEL}' is not available")
            return False
            
        # Step 3: Test a simple chat completion with the model
        logger.info(f"Testing model {OLLAMA_MODEL} with prompt: 'Hi, testing Ollama'")
        try:
            # Use a small prompt to test - just enough to verify it works
            client = httpx.Client(timeout=60.0)  # Longer timeout for model test
            response = client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": "Hi, testing Ollama",
                    "stream": False,
                    "options": {
                        "num_predict": 20,  # Keep this small for faster response
                    }
                }
            )
            
            if response.status_code == 200:
                logger.info(f"Model test successful: {OLLAMA_MODEL}")
                logger.info(f"Ollama is ready with model '{OLLAMA_MODEL}'")
                return True
            else:
                logger.error(f"Model test failed with status code: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Model test failed with error: {str(e)}")
            return False
    
    except Exception as e:
        logger.error(f"Error checking Ollama availability: {str(e)}")
        return False


def check_model_tool_support() -> bool:
    """
    Check if the model supports tool calling
    
    Returns:
        bool: True if the model supports tools, False otherwise
    """
    try:
        # First attempt: Check if the model supports the OpenAI tools format
        # This is the most reliable method but depends on Ollama version
        try:
            client = httpx.Client(timeout=40.0)
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather forecast",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }]
            }
            
            response = client.post(
                f"{OLLAMA_BASE_URL}/api/chat/completions",
                json=payload,
                timeout=40.0
            )
            
            if response.status_code == 200:
                logger.info(f"Model {OLLAMA_MODEL} supports OpenAI tools format")
                return True
            else:
                logger.warning(f"Model {OLLAMA_MODEL} does not support tools via /api/chat/completions: {response.status_code} {response.text}")
        except Exception as e:
            logger.warning(f"Error testing OpenAI tools format: {str(e)}")
        
        # Second attempt: Check if model supports basic JSON mode/structured output
        # This is more widely supported even in older Ollama versions
        try:
            client = httpx.Client(timeout=40.0)
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": """
                Return a JSON object with the following structure:
                {
                  "name": "John",
                  "age": 30,
                  "is_student": false
                }
                Only return the JSON object, nothing else.
                """,
                "format": "json",  # Request JSON output if supported
                "options": {
                    "num_predict": 100
                }
            }
            
            response = client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=40.0
            )
            
            if response.status_code == 200:
                response_text = response.json().get("response", "")
                # Try to parse as JSON to verify it's valid
                try:
                    json_response = json.loads(response_text)
                    if isinstance(json_response, dict) and "name" in json_response:
                        logger.info(f"Model {OLLAMA_MODEL} supports basic structured responses")
                        return True  # Model can format structured output
                except json.JSONDecodeError:
                    logger.warning(f"Model {OLLAMA_MODEL} returned invalid JSON")
            
            logger.warning(f"Model {OLLAMA_MODEL} does not appear to support structured outputs")
            return False
            
        except Exception as e:
            logger.error(f"Error testing structured output: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error checking tool support: {str(e)}")
        return False


def get_ollama_chat_completion(messages: List[Dict[str, str]], stream: bool = False) -> Union[str, Dict[str, Any]]:
    """
    Get a chat completion from Ollama with improved error handling and timeouts
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        stream: Whether to stream the response
        
    Returns:
        Either the assistant's response as a string, or a stream object
    """
    # Large model responses can take time - use a generous timeout
    # Requests typically time out faster on mobile devices, so we need a longer timeout
    timeout = 60.0  # 60 second timeout
    
    try:
        logger.info(f"Sending {len(messages)} messages to Ollama")
        
        # Calculate approximate token count based on characters for better timeout handling
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        # Use a heuristic: ~4 chars per token for English text
        approx_tokens = total_chars / 4
        
        # If input is large, increase timeout
        if approx_tokens > 1000:
            timeout = 90.0  # 90 seconds for larger inputs
            
        # Dynamically adjust max_tokens based on input size
        # This helps prevent timeouts by reducing the expected output size for large inputs
        response_tokens = MAX_TOKENS
        if approx_tokens > 2000:
            response_tokens = min(MAX_TOKENS, 1000)  # Limit output for very large inputs
            
        # Prepare the payload
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_predict": response_tokens
            }
        }
        
        # Attempt the API call with retry logic
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"API attempt {attempt} with timeout={timeout}s")
                
                if stream:
                    # Streaming
                    client = httpx.Client(timeout=timeout)
                    response = client.post(
                        f"{OLLAMA_BASE_URL}/api/chat",
                        json=payload,
                        stream=True
                    )
                    return response
                else:
                    # Non-streaming - use httpx for better timeout handling
                    client = httpx.Client(timeout=timeout)
                    response = client.post(
                        f"{OLLAMA_BASE_URL}/api/chat",
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        response_json = response.json()
                        return response_json["message"]["content"]
                    else:
                        logger.error(f"API call failed with status {response.status_code}: {response.text}")
                        if attempt < max_attempts:
                            logger.info(f"Retrying in 3 seconds...")
                            time.sleep(3)
                        else:
                            raise Exception(f"API call failed with status {response.status_code}: {response.text}")
                            
            except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                logger.error(f"Timeout error on attempt {attempt}: {str(e)}")
                if attempt < max_attempts:
                    logger.info(f"Retrying with longer timeout...")
                    timeout *= 1.5  # Increase timeout for next attempt
                    time.sleep(3)
                else:
                    raise Exception(f"Connection timed out after {max_attempts} attempts")
                    
            except Exception as e:
                logger.error(f"Error on attempt {attempt}: {str(e)}")
                if attempt < max_attempts:
                    logger.info("Retrying in 3 seconds...")
                    time.sleep(3)
                else:
                    raise
                    
        raise Exception(f"Failed to get response after {max_attempts} attempts")
                
    except Exception as e:
        logger.error(f"Exception in chat completion: {str(e)}")
        raise