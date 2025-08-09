import os
import logging
import uuid
import io
import base64 # Added for image encoding
import requests # Added for direct API calls
import csv # Added for CSV handling
import re
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

# Change the import from OllamaLLM to ChatOllama, karena import ChatOllama dipakai untuk chatbase conversation
from langchain_ollama import ChatOllama 

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'
)
# --- DECORATOR UNTUK LOGGING ---
def log_function_calls(func):
    """
    Decorator ini untuk mencatat awal, akhir, dan waktu eksekusi dari sebuah fungsi.
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

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Cek upload file folder dan buat jika tidak ada
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    logging.info(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")

def allowed_file(filename):
    """
    cek file apakah memiliki ekstensi yang diizinkan.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Ollama Model Initialization ---
# bisa menggunakan model multimodal seperti "gemma3:4b" atau "llava", syntax : ollama run "model_name" (e.g., "ollama run llava").
# Multimodal berarti model bisa memproses lebih dari satu jenis masukan (misalnya teks + gambar). project ini requirement nya kecil (bisa menggunakan model monodal text only)
try:
    model = ChatOllama(model="gemma3:4b")
    logging.info("ChatOllama model initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize ChatOllama: {e}")
    model = None # Set model to None if initialization fails

# --- Chat History Management (In-memory, resets on server restart) ---
# Example: [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi there!'}]
# menyimpan riwayat chat dalam memori, akan di-reset saat server restart
chat_history = []
logging.info("Chat history initialized.")

# --- API Configuration ---
GEMINI_API_KEY = 'AIzaSyCLo4Nlbo9_b7SyqZZpTeZL8l7hRW56ROk'
GEMINI_MODEL_NAME = 'gemma-3-27b-it'

ROBOFLOW_API_KEY = '5Wor1dW00UFHynXvJ6do'
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
    reader_ocr = easyocr.Reader(['en'], gpu=True)  # Set gpu=True if you have a compatible GPU
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
def extract_markdown_table(raw_text: str) -> str | None:
    """
    Mengekstrak tabel berformat Markdown dari string mentah yang mungkin berisi teks tambahan.
    Metode ini sangat andal untuk membersihkan output AI.
    """
    start_marker = "REFINED_TABLE_START"
    end_marker = "REFINED_TABLE_END"

    try:
        # Metode 1: Mencari berdasarkan penanda yang spesifik
        start_index = raw_text.find(start_marker)
        end_index = raw_text.find(end_marker)

        if start_index != -1 and end_index != -1:
            # Ekstrak teks di antara penanda
            table_str = raw_text[start_index + len(start_marker):end_index]
            return table_str.strip()

        # Metode 2: Fallback jika penanda tidak ditemukan, cari header tabel
        lines = raw_text.splitlines()
        table_start_line = -1
        for i, line in enumerate(lines):
            clean_line = line.strip()
            if clean_line.startswith('|') and clean_line.endswith('|') and clean_line.count('|') > 2:
                table_start_line = i
                break
        
        if table_start_line != -1:
            table_lines = lines[table_start_line:]
            return "\n".join(table_lines).strip()

        # Jika kedua metode gagal, kembalikan None
        logging.warning("Tidak dapat menemukan tabel Markdown yang valid dalam output.")
        return None

    except Exception as e:
        logging.error(f"Terjadi error saat mengekstrak tabel Markdown: {e}")
        return None
    
def preprocess_image_for_ocr(image_path, target_width=1200):
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
    api_url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}"

    # Get image bytes and encode it as a base64 string
    with open(image_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    # Set up the request parameters and headers
    params = { "api_key": ROBOFLOW_API_KEY }
    headers = { "Content-Type": "application/x-www-form-urlencoded" }
    try:
        response = requests.post( #POST request
            api_url,
            data=img_base64,
            headers=headers,
            params=params
        )
        response.raise_for_status()  # Raises HTTPError, if one occurred for bad status codes (4xx or 5xx)
        
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
"Test Case ID","PIC","Feature","Test Scenario","Test Steps","Data","Expected Result","Actual Result","Status","Notes"
Here is the data from the processed flowchart:
Text Data from Flowchart Components (OCR):
{ocr_data if ocr_data.strip() else "No text data from OCR."}

Bounding Box and Label Information from Object Detection (Roboflow):
{roboflow_data_str if roboflow_data_str.strip() else "No object detection data from Roboflow."}

Critical Instructions:
1. Analyze both the OCR text and Roboflow bounding box data to understand the flowchart structure and logic.
2. Generate only CSV data.
3. For "Test Case ID", use the format "TC-XXX"
4. For "Test Steps", use line breaks between each step (one action per line).
5. Set the following fields as instructed: "Data", "Actual Result", "PIC" leave empty, and "Status" set to "Not Yet".
6. Generate the test case content (like Test Scenario, Test Steps, Expected Result) based on your understanding of the flow. 
7. Use Same Language: Generate ALL the test case, in the **exact same language** you detected. If the input is in Bahasa Indonesia, your entire output must also be in Bahasa Indonesia.
Your CSV output:
"""
# Critical Instructions:
# 1. Analyze both the OCR text and the Roboflow bounding box data to understand the flow.
# 2. Generate only CSV data.
# 3. For "Test Case ID", use the format "TC-XXX" and For "Steps" use line breaks.
# 4. Leave "Actual Result","PIC", and "Data" **empty**, and set "Status" to "Not Yet".
# 5. Generate the test case content (like Test Scenario, Steps, etc.) 
# 6. Use Same Language: Generate ALL the test case, in the **exact same language** you detected. If the input is in Bahasa Indonesia, your entire output must also be in Bahasa Indonesia.

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

@app.route('/refinement.html', methods=['GET', 'POST'])
@log_function_calls
def refinement():
    def dataframe_to_markdown_table(df):
        header = "| " + " | ".join(df.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        body_rows = [
            "| " + " | ".join(map(str, row)) + " |"
            for row in df.itertuples(index=False)
        ]
        return "\n".join([header, separator, *body_rows])

    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if not (file.filename.endswith('.xlsx') or file.filename.endswith('.csv')):
            return jsonify({'error': 'Unsupported file format.'}), 400

        try:
            if file.filename.endswith('.xlsx'):
                df_original = pd.read_excel(file)
            else: 
                csv_data = io.StringIO(file.stream.read().decode("UTF-8"))
                df_original = pd.read_csv(csv_data)

            if df_original.empty:
                return jsonify({'error': 'The uploaded file is empty.'}), 400
            
            original_test_cases_md = dataframe_to_markdown_table(df_original)
            original_count = len(df_original)

            if not GEMINI_API_KEY:
                return jsonify({'error': 'Gemini API key is not configured.'}), 500

            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            
            prompt = f"""
            You are an expert Software Quality Assurance engineer specializing in test case optimization.
            Your task is to analyze, clean, and enhance the provided test cases.

            Here is the original set of test cases in Markdown format:
            {original_test_cases_md}

            **Your Instructions:**

            1.  **Analyze and Clean:**
                * Review all original test cases to identify any that are **redundant, duplicates, or obsolete**.
                * Do not remove main feature even if common and not detail
                * Remove these identified test cases from the final table.

            2.  **Enhance Coverage:**
                * Based on the remaining test cases, add **new test cases** to improve coverage, focusing on **negative scenarios** and **edge cases**.

            3.  **Structure Your Output (CRITICAL):**
                * You MUST structure your entire response in two distinct parts using the specified markers.
                * Use Same Language: Generate ALL of your output, including the deletion summary and all new test case content (scenarios, steps, notes), in the **exact same language** you detected. If the input is in Bahasa Indonesia, your entire output must also be in Bahasa Indonesia.
                
                * **Part 1: Deletion Summary:**
                    * Begin this section with the exact line `DELETION_SUMMARY_START`.
                    * For each test case you removed, provide a bullet point with its "Test Case ID" and a clear reason for its removal.
                    * If you just update the test case you can add the details with its "Test Case ID" and a clear reason.
                    * End this section with the exact line `DELETION_SUMMARY_END`.
                    * If no test cases were removed, leave this section empty.

                * **Part 2: Refined Test Case Table:**
                    * Begin this section with the exact line `REFINED_TABLE_START`.
                    * Provide the final list of test cases as a **Markdown table**.
                    * This table MUST include a **new final column** named `"Refinement Status"`.
                    * Populate the `"Refinement Status"` column with `'Original'` for existing cases or `'Added'` for new ones or `'Updated'` if the test case was modified.
                    * End this section with the exact line `REFINED_TABLE_END`.

            **Example Output Structure:**

            DELETION_SUMMARY_START
            - **TC-005 (Removed):** This test case is a duplicate of TC-002 and replace with new number TC-003.
            - **TC-008 (Removed):** This scenario is obsolete due to the recent UI update.
            DELETION_SUMMARY_END

            REFINED_TABLE_START
            | Test Case ID | Test Scenario | ... | Refinement Status |
            |---|---|---|---|
            | TC-001 | Verify successful login | ... | Original |
            | TC-009 | Verify login with invalid password | ... | Added |
            | TC-009 | Verify login with invalid password using brute force | ... | Updated |
            REFINED_TABLE_END
            """
            response = model.generate_content(prompt)
            raw_response_text = response.text.strip()
            logging.debug(f"Full raw response from AI:\n{raw_response_text}")

            # Panggil fungsi ekstraktor untuk mendapatkan tabel yang bersih
            clean_markdown_table = extract_markdown_table(raw_response_text)
            
            if not clean_markdown_table:
                # Jika tabel tidak ditemukan setelah pembersihan, kembalikan error
                logging.error("AI did not return a valid table after cleaning.")
                return jsonify({
                    "error": "AI tidak mengembalikan tabel yang valid. Silakan coba lagi.",
                    "raw_response": raw_response_text # Kirim respons asli untuk debug
                }), 500

            # Ekstrak ringkasan penghapusan
            deletion_summary_html = ""
            summary_match = re.search(r'DELETION_SUMMARY_START(.*?)DELETION_SUMMARY_END', raw_response_text, re.DOTALL)
            if summary_match:
                summary_text = summary_match.group(1).strip()
                if summary_text:
                    deletion_summary_html = "<ul>"
                    for line in summary_text.split('\n'):
                        if line.strip().startswith('-'):
                            deletion_summary_html += f"<li>{line.strip()[1:].strip()}</li>"
                    deletion_summary_html += "</ul>"
            
            logging.debug(f"Cleaned Markdown Table:\n{clean_markdown_table}")
            
            return jsonify({
                "refined_markdown": clean_markdown_table, # Gunakan tabel yang sudah bersih ke FE
                "deletion_summary_html": deletion_summary_html,
                "original_count": original_count
            })

        except Exception as e:
            logging.error(f"An error occurred during test case refinement: {e}", exc_info=True)
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

    return render_template('refinement.html')

# --- New Chat API Endpoint ---
@app.route('/chat', methods=['POST'])
@log_function_calls
def chat():
    """
    Handles chat messages and image uploads, interacts with the Ollama LLM,
    and maintains chat history.
    """
    global chat_history
    # Initial Prompt for QA Tester Gemma - prompt ini mendefine persona and guidelines untuk model LLM.
    initial_prompt = """
    You are "Gemma-QA", an expert Software Quality Assurance assistant. Your role is to provide clear, accurate, and concise answers to questions about software testing methodologies, tools, and best practices. Always maintain a professional and helpful tone when responding to user queries.

    Here are instructions from the user outlining your goals and how you should respond:
    This Gemma-QA will act as a Quality Assurance (QA) Tester, focusing on providing guidance, suggestions, and insights related to software quality assurance and testing procedures. Its primary role is to assist users in understanding and navigating the complexities of software testing, including test design, execution, and reporting. It will offer information on various testing methodologies, best practices, tools, and techniques relevant to QA testing.

    The Gemma-QA should maintain a tone that is informative and aligned with quality assurance testing principles. It should provide accurate and detailed responses, emphasize the importance of thorough testing, and encourage best practices in software quality assurance. The Gemma-QA should not provide incorrect or misleading information about QA testing and should not engage in topics outside the realm of software testing and quality assurance. Use Same Language: Generate ALL of your output in the **exact same language** you detected. If the input is in Bahasa Indonesia, your entire output must also be in Bahasa Indonesia

    The Gemma-QA should aim to clarify user queries whenever necessary, providing detailed and comprehensive answers. It should ask for additional details if the user's query is unclear or lacks specific information needed to give a precise response. The Gemma-QA's responses should be tailored to reflect the professionalism and precision expected in the QA testing field.
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
        return jsonify({"response": f"Error: {str(e)}. Ensure Ollama server is running and LLM model is downloaded."}), 500

if __name__ == '__main__':
    app.run(debug=True)