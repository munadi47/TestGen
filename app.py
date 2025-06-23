import os
import logging
import uuid
import io
import base64 # Added for image encoding
import requests # Added for direct API calls
import csv # **FIX**: Ditambahkan untuk parsing CSV yang lebih andal

from flask import Flask, render_template, request, jsonify, url_for, send_file
from werkzeug.utils import secure_filename

# --- Other imports ---
import cv2
import easyocr
import pandas as pd
import numpy as np
import google.generativeai as genai

from PIL import ImageChops
# --- logging ---
import time # Diperlukan untuk mengukur waktu eksekusi
import functools # Diperlukan untuk decorator

# Change the import from OllamaLLM to ChatOllama
from langchain_ollama import ChatOllama 

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'
)
# --- DECORATOR UNTUK LOGGING ---
def log_function_calls(func):
    """
    Decorator ini mencatat awal, akhir, dan waktu eksekusi
    dari sebuah fungsi.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Memulai eksekusi fungsi: {func.__name__}...")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logging.error(f"Terjadi error pada fungsi {func.__name__}: {e}", exc_info=True)
            raise # Melemparkan kembali exception setelah logging
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"Selesai eksekusi fungsi: {func.__name__}. Durasi: {duration:.4f} detik.")
    return wrapper

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
    model = ChatOllama(model="minicpm-v:8b")
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
GEMINI_API_KEY = 'xxx'
GEMINI_MODEL_NAME = 'gemma-3-27b-it'

ROBOFLOW_API_KEY = 'xxx'
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
@log_function_calls
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
@log_function_calls
def chatbot():
    return render_template('chatbot.html')

# @app.route('/refinement.html')
# def refinement():
#     return render_template('refinement.html')
@app.route('/refinement.html', methods=['GET', 'POST'])
@log_function_calls
def refinement():
    # **FIX**: Moved the function definition inside the route to guarantee its availability.
    def dataframe_to_markdown_table(df):
        """
        Converts a pandas DataFrame to a simple Markdown table string.
        """
        header = "| " + " | ".join(df.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        body_rows = []
        for row in df.itertuples(index=False):
            # Ensure all items are converted to string to avoid errors
            body_rows.append("| " + " | ".join(map(str, row)) + " |")
        body = "\n".join(body_rows)
        return "\n".join([header, separator, body])

    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not (file.filename.endswith('.xlsx') or file.filename.endswith('.csv')):
            return jsonify({'error': 'Unsupported file format. Please upload an .xlsx or .csv file.'}), 400

        try:
            if file.filename.endswith('.xlsx'):
                df_original = pd.read_excel(file)
            else: 
                csv_data = io.StringIO(file.stream.read().decode("UTF-8"))
                df_original = pd.read_csv(csv_data)

            if df_original.empty:
                return jsonify({'error': 'The uploaded file is empty or could not be read.'}), 400
            
            # Convert original dataframe to Markdown to be sent to the AI
            original_test_cases_md = dataframe_to_markdown_table(df_original)
            original_count = len(df_original)

            if not GEMINI_API_KEY or GEMINI_API_KEY == 'YOUR_GEMINI_API_KEY':
                return jsonify({'error': 'Gemini API key is not configured on the server.'}), 500

            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            
            prompt = f"""
            You are a Software Quality Assurance expert specializing in test case refinement.
            Your task is to analyze the provided test cases and improve the test suite coverage.

            **Instructions:**
            1.  **Analyze Existing Cases:** Carefully review the provided test cases in the Markdown table format below.
            2.  **Enhance Coverage:** Add new rows to the table to cover missing scenarios. This should include:
                * **Negative Test Cases:** Scenarios with invalid inputs or actions that should result in an error.
                * **Edge Cases:** Scenarios that test the boundaries of input values.
            3.  **Maintain Format:** Preserve the exact same Markdown table format as the input.
            4.  **Do Not Delete:** Return all original rows plus your new ones in a single Markdown table.
            5.  **Output Format:** The final output MUST be only a Markdown table. Do not add any explanations, introductions, or markdown formatting like ```.

            **Original Test Cases (Markdown Table):**
            {original_test_cases_md}

            **Your Refined Markdown Table Output:**
            """
            
            response = model.generate_content(prompt)
            
            # Clean the response text from the model
            refined_markdown_text = response.text.strip()
            if refined_markdown_text.startswith("```markdown"):
                refined_markdown_text = refined_markdown_text[len("```markdown"):].strip()
            if refined_markdown_text.endswith("```"):
                refined_markdown_text = refined_markdown_text[:-3].strip()

            return jsonify({
                "refined_markdown": refined_markdown_text,
                "original_count": original_count
            })

        except Exception as e:
            logging.error(f"An error occurred during test case refinement: {e}")
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

    return render_template('refinement.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     return jsonify({"response": "Chatbot functionality is separate."})

# --- New Chat API Endpoint ---
@app.route('/chat', methods=['POST'])
@log_function_calls
def chat():
    """
    Handles chat messages and image uploads, interacts with the Ollama LLM,
    and maintains chat history.
    """
    global chat_history
    # --- Initial Prompt for QA Tester Gemma ---
    # This prompt defines the persona and guidelines for the LLM.
    initial_prompt = """
    You are a "Gemma" – a version of ChatGemma that has been customized for a specific use case. 
    Gemmas use custom instructions, capabilities, and data to optimize ChatGemma for a more narrow set of tasks. 
    You yourself are a Gemma created by a user, and your name is QA Tester. 
    Note: Gemma is also a technical term in AI, but in most cases if the users asks you about Gemmas assume they are referring to the above definition.

    Here are instructions from the user outlining your goals and how you should respond:
    This Gemma will act as a Quality Assurance (QA) Tester, focusing on providing guidance, suggestions, and insights related to software quality assurance and testing procedures. Its primary role is to assist users in understanding and navigating the complexities of software testing, including test design, execution, and reporting. It will offer information on various testing methodologies, best practices, tools, and techniques relevant to QA testing.

    The Gemma should maintain a tone that is informative and aligned with quality assurance testing principles. It should provide accurate and detailed responses, emphasize the importance of thorough testing, and encourage best practices in software quality assurance. The Gemma should not provide incorrect or misleading information about QA testing and should not engage in topics outside the realm of software testing and quality assurance.

    The Gemma should aim to clarify user queries whenever necessary, providing detailed and comprehensive answers. It should ask for additional details if the user's query is unclear or lacks specific information needed to give a precise response. The Gemma's responses should be tailored to reflect the professionalism and precision expected in the QA testing field.
    """

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
            current_user_message_parts.append({"image": filepath})
        except Exception as e:
            logging.error(f"Error saving image: {e}")
            return jsonify({"response": f"Error saving image: {str(e)}"}), 500
    
    if not user_input and not image_file:
        return jsonify({"response": "Please enter a message or upload an image."}), 400

    # Add the initial prompt as a system message if chat history is empty
    # This ensures the model's persona is set at the beginning of the conversation.
    if not chat_history:
        chat_history.append({'role': 'system', 'content': initial_prompt})
        logging.info("Initial system prompt added to chat history.")

    # Append current user message to global chat history
    chat_history.append({'role': 'user', 'content': current_user_message_parts})

    # Prepare messages for the ChatOllama model
    langchain_messages = []
    for msg in chat_history:
        if isinstance(msg['content'], list): # Multimodal content
            langchain_content_parts = []
            for part in msg['content']:
                if 'text' in part:
                    langchain_content_parts.append({"type": "text", "text": part['text']})
                if 'image' in part:
                    # For Ollama's multimodal models, file paths are typically sent as local file URLs
                    langchain_content_parts.append({"type": "image_url", "image_url": {"url": f"file://{os.path.abspath(part['image'])}"}} )
            langchain_messages.append({'role': msg['role'], 'content': langchain_content_parts})
        else: # Text-only content (including the initial system prompt and assistant responses)
            langchain_messages.append({'role': msg['role'], 'content': msg['content']})

    logging.debug(f"Messages sent to ChatOllama model: {langchain_messages}")

    try:
        response_message = model.invoke(langchain_messages)
        response_text = response_message.content 
        logging.debug(f"ChatOllama raw response content: {response_text}")

        # Append bot response to global chat history
        chat_history.append({'role': 'assistant', 'content': response_text})
        logging.debug(f"Updated chat history: {chat_history}")

        return jsonify({"response": response_text})
    except Exception as e:
        logging.error(f"Error invoking ChatOllama model: {e}")
        return jsonify({"response": f"Error: {str(e)}. Ensure Ollama server is running and model ('llava') is downloaded."}), 500

if __name__ == '__main__':
    app.run(debug=True)
