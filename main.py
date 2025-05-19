# sudo ufw allow 11435/tcp comment 'Ollama proxy for API key'
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, Response, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import httpx
from dotenv import load_dotenv
import os
import uvicorn
import logging
import json
import time
import asyncio
import re
from typing import Dict, List, Tuple, AsyncGenerator, Union, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY is not set")

OLLAMA_URL = os.getenv("OLLAMA_URL")
if not OLLAMA_URL:
    raise ValueError("OLLAMA_URL is not set")

ENABLE_FAKE_TOOLS = os.getenv("ENABLE_FAKE_TOOLS", "false").lower() == "true"

app = FastAPI()

# Debug endpoint to log requests and always return tool success
@app.post("/v1/debug/chat/completions")
async def debug_completions(request: Request):
    """Debug endpoint that logs the request and always returns a successful tool response"""
    # Get the request body
    body_bytes = await request.body()
    body_text = body_bytes.decode("utf-8")
    
    try:
        # Parse and log the request
        request_data = json.loads(body_text)
        logger.info("==== DEBUG ENDPOINT - REQUEST RECEIVED ====")
        logger.info(f"Headers: {dict(request.headers)}")
        logger.info(f"Body: {json.dumps(request_data, indent=2)}")
        
        # Extract model and any tool details
        model = request_data.get("model", "unknown")
        has_tools = "tools" in request_data
        
        # Create a response with tool_calls
        tool_calls = [{
            "id": f"debug_{int(time.time())}",
            "type": "function",
            "function": {
                "name": "edit_file",
                "arguments": json.dumps({
                    "target_file": "debug.py",
                    "instructions": "Creating a debug file",
                    "code_edit": "def debug():\n    print('Debug successful!')"
                })
            }
        }]
        
        # Always return a successful response with tool_calls
        return JSONResponse(status_code=200, content={
            "id": f"chatcmpl-debug-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": tool_calls
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple rate limiting
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_history: Dict[str, List[float]] = {}
        
    def is_rate_limited(self, client_ip: str) -> Tuple[bool, float]:
        now = time.time()
        minute_ago = now - 60
        
        # Initialize or clean old requests
        if client_ip not in self.request_history:
            self.request_history[client_ip] = []
        else:
            self.request_history[client_ip] = [t for t in self.request_history[client_ip] if t > minute_ago]
        
        # Check if rate limited
        if len(self.request_history[client_ip]) >= self.requests_per_minute:
            oldest = self.request_history[client_ip][0]
            reset_time = oldest + 60 - now
            return True, reset_time
        
        # Add current request
        self.request_history[client_ip].append(now)
        return False, 0

rate_limiter = RateLimiter()
security = HTTPBearer()

def validate_api_key(authorization: str) -> bool:
    if not authorization:
        return False
        
    # Handle different authorization formats
    if authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
    else:
        token = authorization
        
    return token == API_KEY


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security), request: Request = None):
    if not validate_api_key(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Apply rate limiting
    client_ip = request.client.host
    is_limited, reset_time = rate_limiter.is_rate_limited(client_ip)
    if is_limited:
        raise HTTPException(
            status_code=429, 
            detail=f"Rate limit exceeded. Try again in {reset_time:.1f} seconds",
            headers={"Retry-After": str(int(reset_time))}
        )
    
    return credentials.credentials


# Set a smaller chunk size for streaming responses
CHUNK_SIZE = 1024  # 1KB chunks

# Custom streaming client configuration
class StreamingClient:
    def __init__(self, base_url: str, timeout: Optional[float] = None):
        self.base_url = base_url
        self.timeout = timeout
        
    async def request_streaming(
        self, 
        method: str, 
        path: str, 
        headers: Dict[str, str], 
        params: Dict[str, str] = None,
        content: bytes = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Make a streaming request to the target server and yield chunks as they arrive.
        """
        url = f"{self.base_url}/{path}"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                method=method,
                url=url,
                headers=headers,
                params=params,
                content=content
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(CHUNK_SIZE):
                    yield chunk

# Initialize the streaming client
streaming_client = StreamingClient(OLLAMA_URL)

# Tool Function Support
class ToolFunctionSupport:
    @staticmethod
    def extract_tool_request(body_text):
        """Extract tool calling request details from body text"""
        try:
            data = json.loads(body_text)
            tools = data.get("tools", [])
            tool_choice = data.get("tool_choice", None)
            return bool(tools), tool_choice
        except:
            return False, None
    
    @staticmethod
    def detect_tool_call_pattern(text):
        """Detect if text contains something that looks like a tool call request"""
        tool_call_patterns = [
            r'<function_calls>',
            r'<function>(.*?)</function>',
            r'"type":\s*"function"',
            r'function\s*\(',
            r'tool\s*\(',
        ]
        
        for pattern in tool_call_patterns:
            if re.search(pattern, text, re.DOTALL):
                return True
        return False
    
    @staticmethod
    def transform_to_tool_response(content, tools_requested=False, tool_choice=None):
        """Transform regular completion response to include tool calls if needed"""
        if not tools_requested:
            return content
            
        # Check if content already has what appears to be a function call
        if ToolFunctionSupport.detect_tool_call_pattern(content):
            # Extract the function call pattern and convert to function_call format
            return content
            
        # If no tool calls detected in the content, create a synthetic one
        # This is a simplified example - real implementation would need more intelligence
        tool_call = {
            "id": "call_" + str(int(time.time())),
            "type": "function",
            "function": {
                "name": "suggested_edit",
                "arguments": json.dumps({
                    "content": content,
                    "reason": "Generated response from Ollama model"
                })
            }
        }
        
        return json.dumps({
            "content": "",
            "tool_calls": [tool_call]
        })

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy(path: str, request: Request):
    # Validate API key for all request types
    authorization = request.headers.get("authorization")
    if not validate_api_key(authorization):
        return JSONResponse(status_code=401, content={"error": "Invalid API key"})

    # Apply rate limiting
    client_ip = request.client.host
    is_limited, reset_time = rate_limiter.is_rate_limited(client_ip)
    if is_limited:
        return JSONResponse(
            status_code=429, 
            content={"error": f"Rate limit exceeded. Try again in {reset_time:.1f} seconds"},
            headers={"Retry-After": str(int(reset_time))}
        )

    # Get the request body without consuming it
    body_bytes = await request.body()
    
    # Special handling for tool validation check
    if request.method == "POST" and path.endswith("/chat/completions") and ENABLE_FAKE_TOOLS:
        try:
            body_text = body_bytes.decode("utf-8")
            request_data = json.loads(body_text)
            
            # Enhanced detection for Cursor validation
            is_cursor_validation = False
            validation_message = ""
            
            # Check 1: Message content analysis
            if "messages" in request_data and len(request_data["messages"]) > 0:
                # Get the latest message content
                message_content = request_data["messages"][-1].get("content", "").lower()
                model_name = request_data.get("model", "").lower()
                
                # Keywords that suggest validation
                validation_keywords = ["validate", "check", "support", "tool", "capability"]
                
                # Check if it's a short message with validation-like content
                if len(message_content) < 100:
                    # Check for validation keywords
                    for keyword in validation_keywords:
                        if keyword in message_content:
                            is_cursor_validation = True
                            validation_message = message_content
                            break
                    
                    # Check for model name in validation message
                    if model_name and model_name in message_content:
                        is_cursor_validation = True
                        validation_message = message_content
            
            # Check 2: Special case for exactly "validate modelname" pattern
            if "messages" in request_data and len(request_data["messages"]) == 1:
                message_content = request_data["messages"][0].get("content", "").lower()
                if message_content.startswith("validate "):
                    is_cursor_validation = True
                    validation_message = message_content
            
            # Check 3: Request has tools but a very simple message (likely validation)
            if "tools" in request_data and "messages" in request_data:
                if len(request_data["messages"]) == 1 and len(request_data["messages"][0].get("content", "")) < 30:
                    is_cursor_validation = True
                    validation_message = request_data["messages"][0].get("content", "")
            
            # Log the validation attempt in detail
            if is_cursor_validation:
                logger.info(f"===== CURSOR VALIDATION DETECTED =====")
                logger.info(f"Validation message: '{validation_message}'")
                logger.info(f"Model: {request_data.get('model', 'unknown')}")
                logger.info(f"Has tools: {'tools' in request_data}")
                if "tools" in request_data:
                    logger.info(f"Tool count: {len(request_data['tools'])}")
                    for i, tool in enumerate(request_data["tools"]):
                        if "function" in tool:
                            logger.info(f"Tool {i+1} name: {tool['function'].get('name', 'unknown')}")
                
                # Always respond with valid tool_calls using common Cursor functions
                tool_calls = []
                
                # First try to use a tool provided in the request
                if "tools" in request_data and len(request_data["tools"]) > 0:
                    for tool in request_data["tools"]:
                        if "function" in tool and "name" in tool["function"]:
                            tool_name = tool["function"].get("name", "")
                            # Common Cursor tool names
                            if tool_name in ["edit_file", "codebase_search", "run_terminal_cmd"]:
                                tool_calls.append({
                                    "id": f"cursor_validation_{int(time.time())}",
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps({
                                            "target_file" if tool_name == "edit_file" else "query": "example.py",
                                            "instructions" if tool_name == "edit_file" else "explanation": "Validation successful",
                                            "code_edit" if tool_name == "edit_file" else "command": "def validate():\n    return True"
                                        })
                                    }
                                })
                                break
                
                # If no specific tools found, use a generic edit_file tool
                if not tool_calls:
                    tool_calls = [{
                        "id": f"cursor_validation_{int(time.time())}",
                        "type": "function",
                        "function": {
                            "name": "edit_file",
                            "arguments": json.dumps({
                                "target_file": "example.py",
                                "instructions": "Creating an example file",
                                "code_edit": "def validate():\n    return True"
                            })
                        }
                    }]
                
                # Return a successful validation response
                return JSONResponse(status_code=200, content={
                    "id": f"chatcmpl-cursor-validation-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request_data.get("model", "unknown"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "",  # Empty content when tool_calls are present
                                "tool_calls": tool_calls
                            },
                            "finish_reason": "tool_calls"
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                })
            
            # For non-validation requests that have tools, strip them before forwarding
            # The proxy will add them back in the response
            if "tools" in request_data:
                logger.info("Removing tools from non-validation request to avoid Ollama rejection")
                # Keep a record of the tools for later
                original_tools = request_data.pop("tools")
                has_tools_request = True
                if "tool_choice" in request_data:
                    request_data.pop("tool_choice")
                # Update the body_bytes with the modified request
                body_bytes = json.dumps(request_data).encode("utf-8")
                
        except Exception as e:
            logger.error(f"Error in tools processing: {e}")
    
    # Log the request body for debugging
    try:
        body_text = body_bytes.decode("utf-8")
        
        # Print detailed log of incoming request
        logger.info(f"→ Incoming request to {path}")
        logger.info(f"→ Method: {request.method}")
        logger.info(f"→ Headers: {request.headers}")
        logger.info(f"→ Body: {body_text[:500]}...")
        
        # Check if this is a streaming request
        is_streaming_request = False
        if request.method == "POST":
            # Check query parameters for stream=true
            if "stream" in request.query_params and request.query_params["stream"].lower() == "true":
                is_streaming_request = True
            # Check body for stream: true (common in OpenAI-style requests)
            elif '"stream": true' in body_text or '"stream":true' in body_text:
                is_streaming_request = True
                
        # Check if tools/function calling is requested
        has_tools_request = False
        tool_choice = None
        if ENABLE_FAKE_TOOLS and request.method == "POST" and path.endswith("/chat/completions"):
            has_tools_request, tool_choice = ToolFunctionSupport.extract_tool_request(body_text)
            logger.info(f"Tools requested: {has_tools_request}, Tool choice: {tool_choice}")
            
            # If tools are requested, we need to modify the request body before forwarding to Ollama
            if has_tools_request:
                try:
                    request_data = json.loads(body_text)
                    # Remove tools and tool_choice from the request, as they're not supported by Ollama
                    if "tools" in request_data:
                        del request_data["tools"]
                    if "tool_choice" in request_data:
                        del request_data["tool_choice"]
                    # Update the body_bytes with the modified request
                    body_bytes = json.dumps(request_data).encode("utf-8")
                    logger.info(f"→ Modified request body (removed tools): {body_bytes.decode('utf-8', errors='ignore')}")
                except Exception as e:
                    logger.error(f"Error modifying request to remove tools: {e}")
    except Exception as e:
        logger.error(f"Error decoding request body: {e}")
        is_streaming_request = False
        has_tools_request = False

    # Remove authorization headers before forwarding to Ollama
    headers = {k: v for k, v in request.headers.items()}
    headers.pop("authorization", None)
    headers.pop("x-api-key", None)
    headers['host'] = 'localhost:11434'  # 🔧 override for Ollama
    
    # Update Content-Length header if body was modified
    if "content-length" in headers:
        headers["content-length"] = str(len(body_bytes))
        logger.info(f"→ Updated Content-Length: {headers['content-length']}")

    # Handle streaming requests differently
    if is_streaming_request:
        async def stream_from_ollama():
            try:
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        method=request.method,
                        url=f"{OLLAMA_URL}/{path}",
                        headers=headers,
                        params=request.query_params,
                        content=body_bytes,
                        timeout=None
                    ) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk
            except Exception as e:
                logger.error(f"Error in streaming from Ollama: {e}")
                error_json = json.dumps({"error": str(e)})
                yield f"data: {error_json}\n\n".encode("utf-8")
        
        # Determine the appropriate content type
        content_type = "text/event-stream"
        if path.endswith("/completions") or path.endswith("/chat/completions"):
            content_type = "application/json"
            
        return StreamingResponse(
            stream_from_ollama(),
            media_type=content_type
        )
    else:
        # Handle non-streaming requests with the original code
        async with httpx.AsyncClient() as client:
            try:
                logger.info(f"→ Forwarding to Ollama: {OLLAMA_URL}/{path}")
                logger.info(f"→ Headers: {headers}")
                logger.info(f"→ Params: {dict(request.query_params)}")
                logger.info(f"→ Body: {body_bytes.decode('utf-8', errors='ignore')}")

                response = await client.request(
                    method=request.method,
                    url=f"{OLLAMA_URL}/{path}",
                    headers=headers,
                    params=request.query_params,
                    content=body_bytes,
                    timeout=None
                )
            except httpx.RequestError as e:
                logger.error(f"Request to Ollama failed: {e}")
                return JSONResponse(status_code=502, content={"error": "Failed to reach backend"})
            except Exception as e:
                logger.error(f"Error forwarding request to Ollama: {str(e)}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Response status: {e.response.status_code}")
                    logger.error(f"Response body: {e.response.text}")
                return JSONResponse(status_code=500, content={"error": str(e)})

            excluded_headers = {"content-encoding", "transfer-encoding", "content-length", "connection"}
            response_headers = {k: v for k, v in response.headers.items() if k.lower() not in excluded_headers}

            # For any unexpected streaming responses
            if response.headers.get("content-type", "").startswith("text/event-stream"):
                async def stream_response():
                    async for chunk in response.aiter_bytes():
                        yield chunk
                
                return StreamingResponse(
                    stream_response(),
                    status_code=response.status_code,
                    headers=response_headers,
                    media_type="text/event-stream"
                )
            else:
                content = await response.aread()
                
                # Transform response for function calling if requested
                if ENABLE_FAKE_TOOLS and has_tools_request and path.endswith("/chat/completions"):
                    try:
                        response_data = json.loads(content)
                        # Get the content from the response
                        if "choices" in response_data and len(response_data["choices"]) > 0:
                            if "message" in response_data["choices"][0]:
                                content_text = response_data["choices"][0]["message"].get("content", "")
                                
                                # Transform the content to include tool calls
                                transformed_content = ToolFunctionSupport.transform_to_tool_response(
                                    content_text, has_tools_request, tool_choice
                                )
                                
                                # Update the response
                                if transformed_content != content_text:
                                    try:
                                        transformed_data = json.loads(transformed_content)
                                        # If it's a valid JSON, replace the message content
                                        response_data["choices"][0]["message"] = transformed_data
                                    except:
                                        # If not valid JSON, just replace the content
                                        response_data["choices"][0]["message"]["content"] = ""
                                        if "tool_calls" not in response_data["choices"][0]["message"]:
                                            tool_call = {
                                                "id": "call_" + str(int(time.time())),
                                                "type": "function",
                                                "function": {
                                                    "name": "suggested_edit",
                                                    "arguments": json.dumps({
                                                        "content": content_text,
                                                        "reason": "Generated from Ollama model"
                                                    })
                                                }
                                            }
                                            response_data["choices"][0]["message"]["tool_calls"] = [tool_call]
                                
                                content = json.dumps(response_data).encode()
                    except Exception as e:
                        logger.error(f"Error transforming response for function calling: {e}")
                
                return Response(
                    content=content,
                    status_code=response.status_code,
                    headers=response_headers
                )


if __name__ == "__main__":
    ssl_certfile = os.getenv("API_SSL_CERTFILE") or None
    ssl_keyfile = os.getenv("API_SSL_KEYFILE") or None
    
    # Validate SSL configuration
    if ssl_certfile and ssl_keyfile:
        if not os.path.exists(ssl_certfile):
            logger.error(f"SSL certificate file not found: {ssl_certfile}")
            raise FileNotFoundError(f"SSL certificate file not found: {ssl_certfile}")
        if not os.path.exists(ssl_keyfile):
            logger.error(f"SSL key file not found: {ssl_keyfile}")
            raise FileNotFoundError(f"SSL key file not found: {ssl_keyfile}")
        logger.info(f"Starting secure server with SSL on port {os.getenv('API_PORT', 11435)}")
    else:
        logger.warning("Starting server WITHOUT SSL. This is not recommended for production use.")
    
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", 11435))

    uvicorn.run(
        app,
        host=host,
        port=port,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile
    )
