# voiceover.py
from gtts import gTTS
import os

def generate_voiceover(script_text, output_path="output/voiceover.mp3"):
    """Convert script text to MP3 using Google TTS."""
    # Trim if too long (gTTS has limits, but 500 chars is safe)
    text = script_text[:1000]
    tts = gTTS(text, lang="en", slow=False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tts.save(output_path)
    return output_path