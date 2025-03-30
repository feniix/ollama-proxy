from dotenv import load_dotenv
import os
load_dotenv(override=True)
from openai import OpenAI
import base64

url = f"https://{os.getenv('SSL_DOMAIN')}:11435/v1"
os.environ["OPENAI_API_BASE"] = url
os.environ["OPENAI_BASE_URL"] = url

client = OpenAI(
    base_url = url,
    api_key=os.getenv("API_KEY")
)

# Encode the local image as Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
]

if __name__ == "__main__":

    image_path = "images/image.png"
    base64_image = encode_image(image_path)

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"}
        ]
    })

    response = client.chat.completions.create(
        model="gemma3:27b",
        messages=messages,
        stream=True,
        timeout=60  # Add a timeout to prevent hanging
    )

    full_response = ""
    for chunk in response:
        content = chunk.choices[0].delta.content or ""
        if content:
            print(content, end="", flush=True)
            full_response += content
    print("\n\nStreaming completed successfully!")

    messages.append({"role": "assistant", "content": full_response})