import os
import requests
import logging
import gradio as gr
from dotenv import load_dotenv
from pydub import AudioSegment
from io import BytesIO
import time
import sqlite3

# Load environment variables from config.env
load_dotenv("config.env")

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configure Hugging Face API URL and headers for Meta-Llama-3-70B-Instruct
api_url = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct"
huggingface_api_key = os.getenv("HF_API_TOKEN")
headers = {"Authorization": f"Bearer {huggingface_api_key}"}

# Function to query the Hugging Face model
def query_huggingface(payload):
    logging.debug(f"Querying model with payload: {payload}")
    response = requests.post(api_url, headers=headers, json=payload)
    logging.debug(f"Received response: {response.status_code} {response.text}")
    return response.json()

# Function to query the Whisper model for audio transcription
def query_whisper(audio_path):
    API_URL_WHISPER = "https://api-inference.huggingface.co/models/openai/whisper-large-v2"
    headers = {"Authorization": f"Bearer {huggingface_api_key}"}
    MAX_RETRIES = 5
    RETRY_DELAY = 1  # seconds

    for attempt in range(MAX_RETRIES):
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

            with open(audio_path, "rb") as f:
                data = f.read()

            response = requests.post(API_URL_WHISPER, headers=headers, data=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return {"error": str(e)}

# Function to generate speech from text using Nithu TTS
def generate_speech_nithu(answer):
    API_URL_TTS_Nithu = "https://api-inference.huggingface.co/models/Nithu/text-to-speech"
    headers = {"Authorization": f"Bearer {huggingface_api_key}"}
    payload = {"inputs": answer}
    MAX_RETRIES = 5
    RETRY_DELAY = 1  # seconds

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL_TTS_Nithu, headers=headers, json=payload)
            response.raise_for_status()
            audio_segment = AudioSegment.from_file(BytesIO(response.content), format="flac")
            audio_file_path = "/tmp/answer_nithu.wav"
            audio_segment.export(audio_file_path, format="wav")
            return audio_file_path
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return {"error": str(e)}

# Function to generate speech from text using Ryan TTS
def generate_speech_ryan(answer):
    API_URL_TTS_Ryan = "https://api-inference.huggingface.co/models/espnet/english_male_ryanspeech_fastspeech2"
    headers = {"Authorization": f"Bearer {huggingface_api_key}"}
    payload = {"inputs": answer}
    MAX_RETRIES = 5
    RETRY_DELAY = 1  # seconds

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL_TTS_Ryan, headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()
            audio = response_json.get("audio", None)
            sampling_rate = response_json.get("sampling_rate", None)
            if audio and sampling_rate:
                audio_segment = AudioSegment.from_file(BytesIO(audio), format="wav")
                audio_file_path = "/tmp/answer_ryan.wav"
                audio_segment.export(audio_file_path, format="wav")
                return audio_file_path
            else:
                raise ValueError("Invalid response format from Ryan TTS API")
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return {"error": str(e)}

# Function to fetch patient data from both databases
def fetch_patient_data(cataract_db_path, glaucoma_db_path):
    patient_data = {}

    # Fetch data from cataract_results table
    try:
        conn = sqlite3.connect(cataract_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cataract_results")
        cataract_data = cursor.fetchall()
        conn.close()
        patient_data['cataract_results'] = cataract_data
    except Exception as e:
        patient_data['cataract_results'] = f"Error fetching cataract results: {str(e)}"

    # Fetch data from results table (glaucoma)
    try:
        conn = sqlite3.connect(glaucoma_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM results")
        glaucoma_data = cursor.fetchall()
        conn.close()
        patient_data['results'] = glaucoma_data
    except Exception as e:
        patient_data['results'] = f"Error fetching glaucoma results: {str(e)}"

    return patient_data

# Function to transform fetched data into a readable format
def transform_patient_data(patient_data):
    readable_data = "Readable Patient Data:\n\n"

    if 'cataract_results' in patient_data:
        if isinstance(patient_data['cataract_results'], str):
            readable_data += patient_data['cataract_results'] + "\n"
        else:
            readable_data += "Cataract Results:\n"
            for row in patient_data['cataract_results']:
                if len(row) >= 6:
                    readable_data += f"Patient ID: {row[0]}, Red Quantity: {row[2]}, Green Quantity: {row[3]}, Blue Quantity: {row[4]}, Stage: {row[5]}\n"
                else:
                    readable_data += "Error: Incomplete data row in cataract results\n"
            readable_data += "\n"

    if 'results' in patient_data:
        if isinstance(patient_data['results'], str):
            readable_data += patient_data['results'] + "\n"
        else:
            readable_data += "Glaucoma Results:\n"
            for row in patient_data['results']:
                if len(row) >= 7:
                    readable_data += f"Patient ID: {row[0]}, Cup Area: {row[2]}, Disk Area: {row[3]}, Rim Area: {row[4]}, Rim to Disc Line Ratio: {row[5]}, DDLS Stage: {row[6]}\n"
                else:
                    readable_data += "Error: Incomplete data row in glaucoma results\n"
            readable_data += "\n"

    return readable_data

# Paths to your databases
cataract_db_path = 'cataract_results.db'
glaucoma_db_path = 'glaucoma_results.db'

# Fetch and transform patient data
patient_data = fetch_patient_data(cataract_db_path, glaucoma_db_path)
readable_patient_data = transform_patient_data(patient_data)

# Toggle visibility of input elements based on input type
def toggle_visibility(input_type):
    if (input_type == "Voice"):
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

def cleanup_response(response):
    # Extract only the part after "Answer:" and remove any trailing spaces
    answer_start = response.find("Answer:")
    if answer_start != -1:
        response = response[answer_start + len("Answer:"):].strip()
    return response

def chatbot(audio, input_type, text):
    if input_type == "Voice":
        # Check if audio is a string (file path) or a file object
        if isinstance(audio, str):
            audio_path = audio
        else:
            audio_path = audio.name

        transcription = query_whisper(audio_path)
        if "error" in transcription:
            return "Error transcribing audio: " + transcription["error"], None
        query = transcription['text']
    else:
        query = text

    # Directly use the transformed patient data as context input
    payload = {
        "inputs": f"role: ophthalmologist assistant patient history: {readable_patient_data} question: {query}"
    }

    logging.debug(f"Raw input to the LLM: {payload['inputs']}")

    response = query_huggingface(payload)
    if isinstance(response, list):
        raw_response = response[0].get("generated_text", "Sorry, I couldn't generate a response.")
    else:
        raw_response = response.get("generated_text", "Sorry, I couldn't generate a response.")

    logging.debug(f"Raw output from the LLM: {raw_response}")

    return raw_response, None

# Gradio interface for generating voice response
def generate_voice_response(tts_model, text_response):
    if tts_model == "Nithu (Custom)":
        audio_file_path = generate_speech_nithu(text_response)
        return audio_file_path, None
    elif tts_model == "Ryan (ESPnet)":
        audio_file_path = generate_speech_ryan(text_response)
        return audio_file_path, None
    else:
        return None, None

# Function to update patient history in the interface
def update_patient_history():
    return readable_patient_data
