#!/usr/bin/env python3
import os
import json
import logging
import asyncio
import httpx
import argparse
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

# Default configuration
DEFAULT_API_KEY = os.getenv("API_KEY", "your_custom_api_key")
DEFAULT_PROXY_URL = f"http://{os.getenv('API_HOST', '127.0.0.1')}:{os.getenv('API_PORT', '11435')}"
DEFAULT_MODEL = "llama3"  # Change to match your Ollama model name

# Create a mock Cursor edit_file tool definition
CURSOR_EDIT_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "Edit a file in the workspace",
        "parameters": {
            "type": "object",
            "properties": {
                "target_file": {
                    "type": "string",
                    "description": "Path to the file to modify"
                },
                "instructions": {
                    "type": "string",
                    "description": "Instructions for the edit"
                },
                "code_edit": {
                    "type": "string",
                    "description": "The code changes to make"
                }
            },
            "required": ["target_file", "instructions", "code_edit"]
        }
    }
}

# Create a mock Cursor codebase_search tool definition
CURSOR_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "codebase_search",
        "description": "Search the codebase for relevant code",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "explanation": {
                    "type": "string",
                    "description": "Reason for the search"
                }
            },
            "required": ["query"]
        }
    }
}

async def simulate_cursor_request(client, api_key, proxy_url, model):
    """Simulate a Cursor-style request with tools"""
    logger.info("Simulating a Cursor-style request with tools...")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Typical user task request in Cursor
    test_prompt = """
    I need to add a new function to main.py that prints system information.
    The function should be called print_system_info() and should use the os and platform modules.
    It should print the OS name, version, architecture, and Python version.
    """
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": test_prompt}
        ],
        "tools": [CURSOR_EDIT_FILE_TOOL, CURSOR_SEARCH_TOOL],
        "tool_choice": "auto"
    }
    
    url = f"{proxy_url}/v1/chat/completions"
    
    try:
        response = await client.post(url, headers=headers, json=payload, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"Response status: {response.status_code}")
        
        # Validate structure
        assert "choices" in data, "Response missing 'choices'"
        assert len(data["choices"]) > 0, "No choices in response"
        assert "message" in data["choices"][0], "No message in first choice"
        
        message = data["choices"][0]["message"]
        
        # Check if tool_calls exists
        if "tool_calls" in message:
            logger.info("Tool calls found in response! 🎉")
            logger.info(f"Number of tool calls: {len(message['tool_calls'])}")
            
            for i, tool_call in enumerate(message["tool_calls"]):
                logger.info(f"Tool call {i+1}:")
                logger.info(f"  ID: {tool_call.get('id', 'N/A')}")
                logger.info(f"  Type: {tool_call.get('type', 'N/A')}")
                
                if "function" in tool_call:
                    function_name = tool_call['function'].get('name', 'N/A')
                    logger.info(f"  Function name: {function_name}")
                    
                    # Try to parse arguments as JSON
                    try:
                        args = json.loads(tool_call['function'].get('arguments', '{}'))
                        
                        # Print different information based on the function
                        if function_name == "edit_file":
                            logger.info(f"  Target file: {args.get('target_file', 'N/A')}")
                            logger.info(f"  Instructions: {args.get('instructions', 'N/A')}")
                            code_snippet = args.get('code_edit', '')
                            logger.info(f"  Code edit preview: {code_snippet[:100]}..." if len(code_snippet) > 100 else code_snippet)
                        elif function_name == "codebase_search":
                            logger.info(f"  Search query: {args.get('query', 'N/A')}")
                            logger.info(f"  Explanation: {args.get('explanation', 'N/A')}")
                        else:
                            logger.info(f"  Arguments: {json.dumps(args, indent=2)[:100]}...")
                    except:
                        logger.info(f"  Arguments (raw): {tool_call['function'].get('arguments', '{}')[:100]}...")
            
            return True
        else:
            logger.warning("No tool_calls found in response. Check if ENABLE_FAKE_TOOLS=true in your .env file")
            
            # If no tool_calls but content exists, show that
            if "content" in message:
                content = message["content"]
                logger.info(f"Model content response (first 200 chars): {content[:200]}...")
                
                # Check if the content looks like code
                if "```python" in content or "def print_system_info" in content:
                    logger.info("Response appears to contain code, but not as a tool call")
                    logger.info("This is the typical behavior when ENABLE_FAKE_TOOLS=false")
            
            return False
    except Exception as e:
        logger.error(f"Cursor simulation test failed: {e}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Test Ollama Proxy with Cursor-like requests")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key for the proxy")
    parser.add_argument("--proxy-url", default=DEFAULT_PROXY_URL, help="URL of the Ollama proxy")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to use for testing")
    
    args = parser.parse_args()
    
    logger.info(f"Testing with proxy URL: {args.proxy_url}")
    logger.info(f"Using model: {args.model}")
    
    async with httpx.AsyncClient() as client:
        # Test Cursor-style request
        result = await simulate_cursor_request(client, args.api_key, args.proxy_url, args.model)
        logger.info(f"Cursor simulation test {'passed ✅' if result else 'failed ❌'}")
        
        if not result:
            logger.warning("\nIf the test failed but the proxy is running correctly:")
            logger.warning("1. Make sure you have ENABLE_FAKE_TOOLS=true in your .env file")
            logger.warning("2. Check that your Ollama server is running and accessible")
            logger.warning("3. Verify that the model specified exists in your Ollama installation")

if __name__ == "__main__":
    asyncio.run(main()) 