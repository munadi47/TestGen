import os
import logging
import uuid
import io
import base64 # Added for image encoding
import requests # Added for direct API calls

from flask import Flask, render_template, request, jsonify, url_for, send_file
from werkzeug.utils import secure_filename

# --- Other imports ---
import cv2
import easyocr
import pandas as pd
import numpy as np
import google.generativeai as genai

from PIL import Image
# The 'inference_sdk' import has been removed

# Change the import from OllamaLLM to ChatOllama
from langchain_ollama import ChatOllama 

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Create an instance of the Flask application
app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    logging.info(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")

def allowed_file(filename):
    """
    Checks if a filename has an allowed image extension.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Ollama Model Initialization ---
# IMPORTANT: Ensure your Ollama server is running and the model is downloaded.
# You can download it using: ollama run "model_name" (e.g., "ollama run llava")
try:
    # Initialize the ChatOllama model.
    # ChatOllama correctly handles the list of messages with roles and multimodal content.
    model = ChatOllama(model="llava")
    logging.info("ChatOllama model initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize ChatOllama: {e}")
    model = None # Set model to None if initialization fails

# --- Chat History Management (In-memory, resets on server restart) ---
# This will store the conversation history as a list of dictionaries,
# suitable for Langchain's message format.
# Example: [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi there!'}]
chat_history = []
logging.info("Chat history initialized.")

# --- API Configuration ---
GEMINI_API_KEY = 'AIzaSyBv81a6bSC4SqdkzZ8nej6zlgyqJGhL6Aw'
GEMINI_MODEL_NAME = 'gemma-3-27b-it'

ROBOFLOW_API_KEY = 'wSWpT02lhyi83iIpUmMU'
ROBOFLOW_MODEL_ID = 'flow-chart-detection/2' 

# --- Initializations ---
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    logging.info(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

try:
    logging.info("Initializing EasyOCR Reader...")
    reader_ocr = easyocr.Reader(['en'], gpu=False)
    logging.info("EasyOCR Reader initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing EasyOCR Reader: {e}")
    reader_ocr = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logging.info("Google Generative AI SDK configured successfully.")
    except Exception as e:
        logging.error(f"Error configuring Google Generative AI SDK: {e}")
else:
    logging.warning("GEMINI_API_KEY environment variable not set. Gemini functions will not work.")

if not ROBOFLOW_API_KEY or not ROBOFLOW_MODEL_ID:
    logging.warning("ROBOFLOW_API_KEY or ROBOFLOW_MODEL_ID not set. Roboflow functions will not work.")

# --- Helper Functions ---

def preprocess_image_for_ocr(image_path, target_width=1200):
    # This function remains the same
    try:
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Image could not be read from path: {image_path}")
            return None
        original_height, original_width = img.shape[:2]
        if original_width == 0: return None
        aspect_ratio = original_height / original_width
        target_height = int(target_width * aspect_ratio)
        interpolation = cv2.INTER_AREA if original_width > target_width else cv2.INTER_CUBIC
        resized_img = cv2.resize(img, (target_width, target_height), interpolation=interpolation)
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
        return thresh
    except Exception as e:
        logging.error(f"Error during image preprocessing: {e}")
        return None

def perform_ocr(processed_image_np):
    # This function remains the same
    if reader_ocr is None: return "Error: EasyOCR reader not initialized."
    if processed_image_np is None: return "Error: Invalid processed image for OCR."
    try:
        results = reader_ocr.readtext(processed_image_np)
        ocr_text = "\n".join([result[1] for result in results])
        return ocr_text if ocr_text.strip() else "No text detected by OCR."
    except Exception as e:
        logging.error(f"Error during OCR: {e}")
        return f"Error during OCR: {str(e)}"

# --- UPDATED query_roboflow function ---
def query_roboflow(image_path):
    """
    Sends an image to the Roboflow API for object detection using the requests library.
    """
    if not ROBOFLOW_API_KEY or not ROBOFLOW_MODEL_ID:
        return "Error: Roboflow API Key or Model ID is not configured in environment variables."

    # Construct the Roboflow API URL. Note the model ID is part of the URL.
    api_url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}"

    # Get image bytes and encode it as a base64 string
    with open(image_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    # Set up the request parameters and headers
    params = { "api_key": ROBOFLOW_API_KEY }
    headers = { "Content-Type": "application/x-www-form-urlencoded" }

    try:
        # Make the POST request
        response = requests.post(
            api_url,
            data=img_base64,
            headers=headers,
            params=params
        )
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        result = response.json()
        predictions = result.get('predictions', [])
        if not predictions:
            return "No predictions received from Roboflow."

        # Format the data to be sent to Gemini
        roboflow_data_list = [
            {
                "label": pred.get('class'),
                "confidence": f"{pred.get('confidence', 0):.2f}",
                "x_center": pred.get('x'),
                "y_center": pred.get('y'),
                "width": pred.get('width'),
                "height": pred.get('height')
            } for pred in predictions
        ]
        return str(roboflow_data_list)

    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred with the Roboflow request: {e}")
        return f"Error: Could not connect to Roboflow API. Details: {str(e)}"
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing Roboflow response: {e}")
        return f"Error: An unexpected error occurred: {str(e)}"


def generate_test_cases_with_gemini(ocr_data, roboflow_data_str):
    # This function remains the same
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not configured."
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e:
        return f"Error initializing Gemini model ('{GEMINI_MODEL_NAME}'): {e}."
    prompt = f"""
You are an AI tasked with creating manual test cases from flowchart data.
The REQUIRED CSV header is:
"Test Case ID","Test Scenario","Test Steps","Expected Result","Actual Result","Status","Notes"
Here is the data from the processed flowchart:
Data Teks dari Komponen Flowchart (OCR):

{ocr_data if ocr_data.strip() else "Tidak ada data teks dari OCR."}

Informasi Bounding Box dan Label dari Deteksi Objek (Roboflow):

{roboflow_data_str if roboflow_data_str.strip() else "Tidak ada data deteksi objek dari Roboflow."}

Critical Instructions:
1. Analyze both the OCR text and the Roboflow bounding box data to understand the flow.
2. Generate only CSV data.
3. For "Test Case ID", use the format "TC-XXX".
4. Leave "Actual Result" empty, and set "Status" to "Not Yet".
5. Ensure the output is in Indonesian.
Your CSV output:
"""
    generation_config = genai.types.GenerationConfig(candidate_count=1, temperature=0.3, max_output_tokens=8000)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    try:
        response = model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)
        if response.parts:
            csv_output = response.parts[0].text.strip()
        elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            csv_output = response.candidates[0].content.parts[0].text.strip()
        else: return "Error: No text content generated by Gemini."
        if "```csv" in csv_output:
            csv_output = csv_output.split("```csv")[1].split("```")[0].strip()
        elif csv_output.strip().startswith("```") and csv_output.strip().endswith("```"):
            csv_output = csv_output.strip()[3:-3].strip()
        if not csv_output.lower().startswith('"test case id"'):
            return f"Error: CSV format from Gemini is not as expected. Header not found."
        return csv_output
    except Exception as e:
        logging.error(f"Error contacting Gemini API: {e}")
        return f"Error: An error occurred with Gemini. Details: {str(e)}"

def csv_to_excel_bytes(csv_data_string):
    # This function remains the same
    if not csv_data_string or not csv_data_string.strip(): return None
    try:
        csv_file_like_object = io.StringIO(csv_data_string)
        df = pd.read_csv(csv_file_like_object)
        output_bytes = io.BytesIO()
        with pd.ExcelWriter(output_bytes, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='TestCases')
        output_bytes.seek(0)
        return output_bytes
    except Exception as e:
        logging.error(f"Error converting CSV to Excel bytes: {e}")
        return None

# --- Flask Routes ---
# (The Flask routes section remains exactly the same)
@app.route('/')
@app.route('/dashboard.html')
def dashboard():
    return render_template('dashboard.html')

@app.route('/flowchart.html')
def flowchart():
    return render_template('flowchart.html')

@app.route('/generate_test_cases', methods=['POST'])
def generate_test_cases_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "No selected file or file type not allowed"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    logging.info(f"File '{filename}' saved to '{filepath}'")
    try:
        processed_image = preprocess_image_for_ocr(filepath)
        if processed_image is None: return jsonify({"error": "Failed to preprocess image."}), 500
        ocr_text_result = perform_ocr(processed_image)
        if "Error:" in ocr_text_result: return jsonify({"error": ocr_text_result}), 500
        logging.info("OCR completed successfully.")
        roboflow_data = query_roboflow(filepath)
        if "Error:" in roboflow_data: return jsonify({"error": roboflow_data}), 500
        logging.info("Roboflow query completed successfully.")
        csv_output = generate_test_cases_with_gemini(ocr_text_result, roboflow_data)
        if "Error:" in csv_output: return jsonify({"error": csv_output}), 500
        excel_bytes = csv_to_excel_bytes(csv_output)
        if excel_bytes is None: return jsonify({"error": "Failed to convert CSV to Excel."}), 500
        output_filename = f"test_cases_{os.path.splitext(filename)[0]}.xlsx"
        return send_file(
            excel_bytes,
            as_attachment=True,
            download_name=output_filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
            logging.info(f"Removed temporary file: {filepath}")

@app.route('/chatbot.html')
def chatbot():
    return render_template('chatbot.html')

@app.route('/refinement.html')
def refinement():
    return render_template('refinement.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     return jsonify({"response": "Chatbot functionality is separate."})

# --- New Chat API Endpoint ---
@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles chat messages and image uploads, interacts with the Ollama LLM,
    and maintains chat history.
    """
    global chat_history

    if model is None:
        return jsonify({"response": "Error: LLM model not initialized. Please check server logs."}), 500

    user_input = request.form.get('user_input', '').strip()
    image_file = request.files.get('image')

    current_user_message_parts = []

    if user_input:
        current_user_message_parts.append({"text": user_input})

    if image_file and allowed_file(image_file.filename):
        try:
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(filepath)
            logging.debug(f"Image saved to {filepath}")
            # For ChatOllama, image paths need to be absolute file URLs
            current_user_message_parts.append({"image": filepath})
        except Exception as e:
            logging.error(f"Error saving image: {e}")
            return jsonify({"response": f"Error saving image: {str(e)}"}), 500
    
    if not user_input and not image_file:
        return jsonify({"response": "Please enter a message or upload an image."}), 400

    # Append current user message to global chat history
    # This stores the internal representation of the user's turn
    chat_history.append({'role': 'user', 'content': current_user_message_parts})

    # Prepare messages for the ChatOllama model, including previous history.
    # ChatOllama expects a list of dictionaries where each dictionary represents a message.
    # For multimodal content, the 'content' key is a list of dictionaries.
    langchain_messages = []
    for msg in chat_history:
        if isinstance(msg['content'], list): # Multimodal content (or just structured text)
            langchain_content_parts = []
            for part in msg['content']:
                if 'text' in part:
                    langchain_content_parts.append({"type": "text", "text": part['text']})
                if 'image' in part:
                    # ChatOllama expects image_url with a 'url' key for file paths
                    langchain_content_parts.append({"type": "image_url", "image_url": {"url": f"file://{os.path.abspath(part['image'])}"}} )
            langchain_messages.append({'role': msg['role'], 'content': langchain_content_parts})
        else: # Text-only content (e.g., assistant's previous responses)
            langchain_messages.append({'role': msg['role'], 'content': msg['content']})

    logging.debug(f"Messages sent to ChatOllama model: {langchain_messages}")

    try:
        # Invoke the ChatOllama model with the prepared messages
        # ChatOllama.invoke expects a list of messages (or BaseMessage objects)
        response_message = model.invoke(langchain_messages)
        
        # The response_message from ChatOllama is a BaseMessage object (e.g., AIMessage)
        # We need to extract its content attribute.
        response_text = response_message.content 
        logging.debug(f"ChatOllama raw response content: {response_text}")

        # Append bot response to global chat history
        chat_history.append({'role': 'assistant', 'content': response_text})
        logging.debug(f"Updated chat history: {chat_history}")

        # Send response to the front-end
        return jsonify({"response": response_text})
    except Exception as e:
        logging.error(f"Error invoking ChatOllama model: {e}")
        return jsonify({"response": f"Error: {str(e)}. Ensure Ollama server is running and model ('llava') is downloaded."}), 500



if __name__ == '__main__':
    app.run(debug=True)
