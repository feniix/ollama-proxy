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
DEFAULT_MODEL = "llama3:8b"  # Change to match your Ollama model name

async def test_basic_completion(client, api_key, proxy_url, model):
    """Test basic completion without tools"""
    logger.info("Testing basic completion without tools...")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Hello, what's your name?"}
        ]
    }
    
    url = f"{proxy_url}/v1/chat/completions"
    
    try:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"Response status: {response.status_code}")
        
        # Validate basic structure
        assert "choices" in data, "Response missing 'choices'"
        assert len(data["choices"]) > 0, "No choices in response"
        assert "message" in data["choices"][0], "No message in first choice"
        assert "content" in data["choices"][0]["message"], "No content in message"
        
        content = data["choices"][0]["message"]["content"]
        logger.info(f"Model response: {content[:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"Basic completion test failed: {e}")
        return False

async def test_tool_completion(client, api_key, proxy_url, model):
    """Test completion with tools enabled"""
    logger.info("Testing completion with tools...")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Can you create a simple Python function to calculate fibonacci numbers?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Edit a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            }
        ]
    }
    
    url = f"{proxy_url}/v1/chat/completions"
    
    try:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"Response status: {response.status_code}")
        
        # Validate structure with tools
        assert "choices" in data, "Response missing 'choices'"
        assert len(data["choices"]) > 0, "No choices in response"
        assert "message" in data["choices"][0], "No message in first choice"
        
        message = data["choices"][0]["message"]
        
        # Check if tool_calls exists (should exist if ENABLE_FAKE_TOOLS=true)
        if "tool_calls" in message:
            logger.info("Tool calls found in response! 🎉")
            logger.info(f"Number of tool calls: {len(message['tool_calls'])}")
            
            for i, tool_call in enumerate(message["tool_calls"]):
                logger.info(f"Tool call {i+1}:")
                logger.info(f"  ID: {tool_call.get('id', 'N/A')}")
                logger.info(f"  Type: {tool_call.get('type', 'N/A')}")
                
                if "function" in tool_call:
                    logger.info(f"  Function name: {tool_call['function'].get('name', 'N/A')}")
                    
                    # Try to parse arguments as JSON
                    try:
                        args = json.loads(tool_call['function'].get('arguments', '{}'))
                        logger.info(f"  Arguments: {json.dumps(args, indent=2)[:100]}...")
                    except:
                        logger.info(f"  Arguments (raw): {tool_call['function'].get('arguments', '{}')[:100]}...")
            
            return True
        else:
            logger.warning("No tool_calls found in response. Check if ENABLE_FAKE_TOOLS=true in your .env file")
            
            # If no tool_calls but content exists, show that
            if "content" in message:
                logger.info(f"Model content response: {message['content'][:100]}...")
            
            return False
    except Exception as e:
        logger.error(f"Tool completion test failed: {e}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Test Ollama Proxy fake tools support")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key for the proxy")
    parser.add_argument("--proxy-url", default=DEFAULT_PROXY_URL, help="URL of the Ollama proxy")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to use for testing")
    
    args = parser.parse_args()
    
    logger.info(f"Testing with proxy URL: {args.proxy_url}")
    logger.info(f"Using model: {args.model}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test basic completion
        basic_result = await test_basic_completion(client, args.api_key, args.proxy_url, args.model)
        logger.info(f"Basic completion test {'passed ✅' if basic_result else 'failed ❌'}")
        
        # Test tool completion
        tool_result = await test_tool_completion(client, args.api_key, args.proxy_url, args.model)
        logger.info(f"Tool completion test {'passed ✅' if tool_result else 'failed ❌'}")
        
        # Overall summary
        if basic_result and tool_result:
            logger.info("All tests passed! 🎉")
        elif basic_result:
            logger.warning("Basic test passed, but tool test failed.")
            logger.warning("Make sure ENABLE_FAKE_TOOLS=true in your .env file.")
        else:
            logger.error("Tests failed. Check the proxy configuration and logs.")

if __name__ == "__main__":
    asyncio.run(main()) 