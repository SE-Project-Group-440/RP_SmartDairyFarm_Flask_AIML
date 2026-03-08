from fastapi import FastAPI,UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scripts.rag_query import rag_answer
from gtts import gTTS
from fastapi import FastAPI, UploadFile, File
from google.cloud import speech_v1p1beta1 as speech
import whisper
import subprocess  # make sure this import exists
import traceback
import os
import uuid



os.environ["PATH"] += os.pathsep + r"G:\ffmpeg-8.0.1-essentials_build\bin"

app = FastAPI(title="Sinhala Dairy RAG API")

# Directory to save generated audio files
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Serve audio files
app.mount("/audio_files", StaticFiles(directory=AUDIO_DIR), name="audio_files")

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    audioUri: str 

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 1️⃣ Get model answer
    answer = rag_answer(req.query)

    # 2️⃣ Generate MP3
    filename = f"{uuid.uuid4()}.mp3"
    file_path = os.path.join(AUDIO_DIR, filename)

    try:
        tts = gTTS(text=answer, lang="si")
        tts.save(file_path)
    except Exception as e:
        print("Error generating audio:", e)
        filename = None

    # 3️⃣ Return answer + audio URL
    if filename:
        audio_uri = f"http://10.154.183.24:8002/audio_files/{filename}"  # Use FastAPI IP & port
    else:
        audio_uri = ""

    return {"answer": answer, "audioUri": audio_uri}


# Set GOOGLE_APPLICATION_CREDENTIALS env variable to your service account JSON
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"G:\Research-Chatbot\service_account.json"



@app.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    raw_path = f"temp_{uuid.uuid4()}.wav"
    converted_path = f"converted_{uuid.uuid4()}.wav"

    try:
        print("🔹 Received audio file:", audio.filename, audio.content_type)

        # Save uploaded file
        with open(raw_path, "wb") as f:
            f.write(await audio.read())
        print(f"✅ Saved raw audio to {raw_path}")

        # Convert to LINEAR16 WAV for Google API
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_path,
            "-ar", "44100", "-ac", "1",
            "-c:a", "pcm_s16le", converted_path
        ], check=True)
        print(f"✅ Converted audio to LINEAR16 WAV: {converted_path}")

        # Initialize Google Cloud client
        try:
            client = speech.SpeechClient()
            print("✅ Google Speech client initialized")
        except Exception as e:
            print("❌ Failed to initialize Google Speech client")
            print(traceback.format_exc())
            return {"error": "Failed to initialize Google Speech client"}

        # Read converted audio
        with open(converted_path, "rb") as f:
            content = f.read()
        print(f"✅ Loaded converted audio, size: {len(content)} bytes")

        # Prepare request
        audio_config = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="si-LK",
        )

        # Call API
        print("🔹 Sending request to Google Speech-to-Text API...")
        response = client.recognize(config=config, audio=audio_config)
        print(f"✅ Received response, results count: {len(response.results)}")

        # Combine transcripts
        text = " ".join([result.alternatives[0].transcript for result in response.results])
        print("🔹 Final transcription:", text)

        return {"text": text}

    except subprocess.CalledProcessError as e:
        print("❌ ffmpeg conversion failed")
        print(e)
        return {"error": f"ffmpeg conversion failed: {e}"}

    except Exception as e:
        print("❌ Unexpected error in /stt endpoint")
        print(traceback.format_exc())
        return {"error": str(e)}

    finally:
        for path in [raw_path, converted_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"♻️ Removed temporary file: {path}")
