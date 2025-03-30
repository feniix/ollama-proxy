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

app = FastAPI()

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
    
    # Log the request body for debugging
    try:
        body_text = body_bytes.decode("utf-8")
        
        # Check if this is a streaming request
        is_streaming_request = False
        if request.method == "POST":
            # Check query parameters for stream=true
            if "stream" in request.query_params and request.query_params["stream"].lower() == "true":
                is_streaming_request = True
            # Check body for stream: true (common in OpenAI-style requests)
            elif '"stream": true' in body_text or '"stream":true' in body_text:
                is_streaming_request = True
    except Exception as e:
        logger.error(f"Error decoding request body: {e}")
        is_streaming_request = False

    # Remove authorization headers before forwarding to Ollama
    headers = {k: v for k, v in request.headers.items()}
    headers.pop("authorization", None)
    headers.pop("x-api-key", None)

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
                return Response(
                    content=await response.aread(),
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
