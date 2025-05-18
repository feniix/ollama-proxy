# Ollama API Proxy

This is a simple FastAPI-based reverse proxy for forwarding API requests to an [Ollama](https://ollama.com) server. It ensures that only requests with a valid **API key** are allowed to access the Ollama backend. It supports all HTTP methods and handles both regular and **streaming** responses. If certificate and key files are provided, it will run with **HTTPS**.

---

## Features

- API key protection for all request types
- Rate limiting to prevent abuse
- CORS support for cross-origin requests
- Optional HTTPS with cert/key file configuration
- Robust error handling and logging
- Support image messages for vision models
- Optional function calling support for Ollama models (fake tools support for Cursor)

---

## Prerequisites

- Python 3.8+
- Ollama running and reachable

---

## Usage

```bash
python main.py
```

## Installation

### Clone the repository
```bash
git clone https://github.com/alfredwallace7/ollama-proxy.git
cd ollama-proxy
```

### Create a virtual environment
```bash
python -m venv venv
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Activate the virtual environment

#### Windows
```bash
.venv\Scripts\activate
```

#### Linux/Mac
```bash
source venv/bin/activate
```

## Configuration

### Rename sample.env to .env

```ini
API_KEY=your_custom_api_key
OLLAMA_URL=http://localhost:11434
API_HOST=0.0.0.0
API_PORT=11435
API_SSL_CERTFILE=./cert.pem       # optional
API_SSL_KEYFILE=./key.pem         # optional
SSL_DOMAIN=my_cert_domain.com     # optional
ENABLE_FAKE_TOOLS=false           # optional, set to "true" to enable fake function calling
```

## Example with openai client library

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://localhost:11435/v1",
    api_key="your_custom_api_key"
)

response = client.chat.completions.create(
    model="qwen2.5:3b",
    messages=[
        {"role": "user", "content": "Say hello"}
    ]
)

print(response.choices[0].message.content)
```

## Example with Invoke-RestMethod (Windows)

```powershell
Invoke-RestMethod -Uri "https://my_cert_domain.com:11435/v1/chat/completions" -Method Post -Headers @{"Authorization"="Bearer your_custom_api_key"; "Content-Type"="application/json"} -Body (@{model="qwen2.5:3b"; messages=@(@{role="user"; content="Say hello"})} | ConvertTo-Json -Depth 10)

```

### Linux/Mac

```bash
curl -X POST https://my_cert_domain.com:11435/v1/chat/completions \
     -H "Authorization: Bearer your_custom_api_key" \
     -H "Content-Type: application/json" \
     -d '{"model": "qwen2.5:3b", "messages": [{"role": "user", "content": "Say hello"}]}'
```

## Function Calling Support for Cursor Integration

This proxy includes a feature to add synthetic function calling capabilities to Ollama models, which can enable better integration with tools like Cursor that depend on function calling for advanced features.

When enabled, the proxy will transform responses from Ollama models to include function calls when requested by the client, allowing Cursor to leverage its advanced features like direct file editing.

To enable this feature:

1. Set the `ENABLE_FAKE_TOOLS` environment variable to `true` in your `.env` file
2. When using Cursor, it will now be able to use its advanced file editing features with Ollama models

Example request with tools:

```bash
curl -X POST https://my_cert_domain.com:11435/v1/chat/completions \
     -H "Authorization: Bearer your_custom_api_key" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "qwen2.5:3b", 
       "messages": [{"role": "user", "content": "Update the file to add logging"}],
       "tools": [{"type": "function", "function": {"name": "edit_file", "description": "Edit a file"}}]
     }'
```

This is an experimental feature and may not work perfectly with all tools or in all contexts.

## Systemd Service (recommended for servers)

Create a service file `/etc/systemd/system/ollama-proxy.service`:

```ini
[Unit]
Description=FastAPI Proxy to Ollama
After=network.target

[Service]
User=your_user_name
WorkingDirectory=/path/to/your/app
ExecStart=/path_to_ollama_proxy/venv/bin/uvicorn main:app --host 0.0.0.0 --port 11435
Restart=always

[Install]
WantedBy=multi-user.target
```

Then enable and start it:

```bash
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable --now ollama-proxy.service
```

## Security Notes

- The API key is checked for **all request types** to ensure maximum security
- Rate limiting is applied to prevent abuse (default: 60 requests per minute per IP)
- HTTPS can be enabled by providing a cert and key file
- May not be suitable for production

## Allow the proxy to be reached from the outside (optional)

### Windows

```powershell
netsh advfirewall firewall add rule name="Ollama Proxy" dir=in action=allow protocol=TCP localport=11435
```

### Debian/Ubuntu

```bash
sudo ufw allow 11435/tcp comment 'Ollama proxy'
```

## License

MIT License
