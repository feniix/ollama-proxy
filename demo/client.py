from dotenv import load_dotenv
import os
load_dotenv(override=True)
from openai import OpenAI
import sys

url = f"https://{os.getenv('SSL_DOMAIN')}:11435/v1"
os.environ["OPENAI_API_BASE"] = url
os.environ["OPENAI_BASE_URL"] = url

client = OpenAI(
    base_url = url,
    api_key=os.getenv("API_KEY")
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
]

if __name__ == "__main__":
    while True:
        user_input = input("User: ")

        if user_input.lower() in ["/exit", "/quit", "/q", "/bye", "/goodbye"]:
            break

        messages.append({"role": "user", "content": user_input})
        
        try:
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
        except KeyboardInterrupt:
            print("\n\nStreaming interrupted by user.")
            break
        except Exception as e:
            print(f"\n\nError during streaming: {e}")