# import os
# import uuid
# import subprocess
# import traceback
# import platform
# from google.cloud import speech_v1p1beta1 as speech

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
#     os.getcwd(),
#     "secrets",
#     "fabled-imagery-457304-q6-c87e0c9cda76.json"
# )

# ffmpeg_dir = os.path.join(os.getcwd(), "bin")
# os.environ["PATH"] += os.pathsep + ffmpeg_dir

# def speech_to_text(file_bytes: bytes) -> dict:
#     raw_path = f"temp_{uuid.uuid4()}.wav"
#     converted_path = f"converted_{uuid.uuid4()}.wav"

#     try:
#         with open(raw_path, "wb") as f:
#             f.write(file_bytes)

#         exe_name = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
#         ffmpeg_path = os.path.join(ffmpeg_dir, exe_name)

#         subprocess.run([
#             ffmpeg_path, "-y", "-i", raw_path,
#             "-ar", "44100", "-ac", "1",
#             "-c:a", "pcm_s16le", converted_path
#         ], check=True)

#         # Initialize Google Speech client
#         client = speech.SpeechClient()

#         # Read converted audio
#         with open(converted_path, "rb") as f:
#             content = f.read()

#         audio_config = speech.RecognitionAudio(content=content)
#         config = speech.RecognitionConfig(
#             encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#             sample_rate_hertz=44100,
#             language_code="si-LK",
#         )

#         response = client.recognize(config=config, audio=audio_config)
#         text = " ".join([result.alternatives[0].transcript for result in response.results])
#         print("STT success:", text)  
#         return {"text": text}

#     except subprocess.CalledProcessError as e:
#         return {"error": f"ffmpeg conversion failed: {e}"}
#     except Exception as e:
#         print("STT failed:", type(e), e)
#         return {"error": str(e)}
#     finally:
#         for path in [raw_path, converted_path]:
#             if os.path.exists(path):
#                 os.remove(path)