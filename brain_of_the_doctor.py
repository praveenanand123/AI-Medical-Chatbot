import os
import base64
import io
import json
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import soundfile as sf
import whisper
from groq import Groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Initialize Whisper model for audio transcription
whisper_model = whisper.load_model("base")

def encode_image(image):
    """Convert PIL Image to base64-encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_image

def transcribe_audio(audio):
    """Transcribe audio input using Whisper."""
    if isinstance(audio, tuple):
        sr, data = audio
        temp_audio_path = "temp_audio.wav"
        sf.write(temp_audio_path, data, sr)
    elif isinstance(audio, str):
        temp_audio_path = audio
    else:
        return "Unsupported audio format."

    result = whisper_model.transcribe(temp_audio_path)
    return result["text"]

def analyze_input(text_input=None, image_input=None, audio_input=None, model="meta-llama/llama-4-scout-17b-16e-instruct"):
    """Process inputs and return a formatted Markdown response."""
    messages = []

    # Handle audio input
    if audio_input is not None:
        transcribed_text = transcribe_audio(audio_input)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"{transcribed_text} Please respond in JSON format."}
            ]
        })
    elif text_input:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"{text_input} Please respond in JSON format."}
            ]
        })

    # Handle image input
    if image_input is not None:
        encoded_img = encode_image(image_input)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Please analyze this medical image and respond in JSON format."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}}
            ]
        })

    if not messages:
        return "Please provide a text, image, or audio input."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"}
        )
        response_content = response.choices[0].message.content
        data = json.loads(response_content)

        # Format the response into Markdown
        markdown_response = f"### 🩺 Diagnosis: {data.get('diagnosis', 'N/A')}\n\n"

        symptoms = data.get("symptoms", [])
        if symptoms:
            markdown_response += "**Symptoms:**\n"
            for symptom in symptoms:
                markdown_response += f"- {symptom}\n"
            markdown_response += "\n"

        causes = data.get("possible_causes", [])
        if causes:
            markdown_response += "**Possible Causes:**\n"
            for cause in causes:
                markdown_response += f"- {cause}\n"
            markdown_response += "\n"

        actions = data.get("recommended_actions", [])
        if actions:
            markdown_response += "**Recommended Actions:**\n"
            for action in actions:
                markdown_response += f"- {action}\n"
            markdown_response += "\n"

        return markdown_response.strip()
    except Exception as e:
        return f"An error occurred while processing the response: {str(e)}"
