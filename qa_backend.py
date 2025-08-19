from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re
import json
import os
from openai import OpenAI

# ✅ OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable not set!")

client = OpenAI(api_key=openai_api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QARequest(BaseModel):
    url: str | None = None
    transcript: str | None = None
    count: int = 10

# Extract YouTube video ID
def get_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:watch\?v=)([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Split transcript into chunks
def chunk_text(text: str, max_chunk_size=3000) -> list[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        if current_size + len(word) > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

@app.post("/generate_qa")
async def generate_qa(request: QARequest):
    count = request.count
    transcript_text = ""

    if count < 1 or count > 50:
        raise HTTPException(status_code=400, detail="Question count must be between 1 and 50")

    # 1️⃣ If user provides transcript directly
    if request.transcript:
        transcript_text = request.transcript.strip()

    # 2️⃣ If user provides YouTube URL
    elif request.url:
        video_id = get_video_id(request.url.strip())
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL format")

        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'bn', 'hi'])
            transcript_text = " ".join([entry['text'] for entry in transcript])
        except (TranscriptsDisabled, NoTranscriptFound):
            raise HTTPException(status_code=400, detail="Transcript not available for this video")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not fetch transcript: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Provide either a YouTube URL or transcript text")

    # Ensure transcript is long enough
    if len(transcript_text.strip()) < 100:
        raise HTTPException(status_code=400, detail="Transcript too short to generate meaningful questions")

    # Process first chunk
    chunks = chunk_text(transcript_text)
    text_to_process = chunks[0]

    # OpenAI prompt
    prompt = f"""Based on the following transcript, generate exactly {count} educational question-answer pairs in JSON format.

Requirements:
- Return ONLY a valid JSON array
- Each question should be clear and educational
- Each answer should be concise but complete
- Focus on key concepts and important information

Format:
[
  {{"question": "What is...?", "answer": "The answer is..."}}
]

Transcript:
{text_to_process}"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an educational content generator. Always return valid JSON arrays only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

    result = response.choices[0].message.content.strip()

    # Validate JSON response
    try:
        json_result = json.loads(result)
        if isinstance(json_result, list):
            return {"result": json_result, "count": len(json_result)}
        else:
            raise HTTPException(status_code=500, detail="Invalid response format from AI")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned invalid JSON format")

@app.get("/health")
async def health_check():
    return {"status": "Server is running", "openai_configured": bool(openai_api_key)}
