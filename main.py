from dotenv import load_dotenv
load_dotenv()
import os
print("🔐 OpenAI Key Found:", os.getenv("OPENAI_API_KEY") is not None)