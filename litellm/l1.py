from litellm import completion
import os

from pathlib import Path
from dotenv import load_dotenv

# Keys live in Agent_Firewall/.env (never commit them). See .env.example.
load_dotenv(Path(__file__).resolve().parents[1] / "Agent_Firewall" / ".env")
response = completion(
    model="groq/openai/gpt-oss-20b", 
    messages=[
       {"role": "user", "content": "What makes China the possible competitor of USA."}
   ],
   stream=True
)
print(response)


# Text
# messages=[{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]

# # Image
# messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}]}]

# # Audio
# messages=[{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}]}]

# # Video
# messages=[{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}]}]

# # File
# messages=[{"role": "user", "content": [{"type": "file", "file": {"file_id": "https://example.com/doc.pdf"}}]}]

# # Document
# messages=[{"role": "user", "content": [{"type": "document", "source": {"type": "text", "media_type": "application/pdf", "data": "<base64>"}}]}]
