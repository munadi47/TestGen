import os
import logging
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from langchain_ollama import ChatOllama

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Create an instance of the Flask application
app = Flask(__name__)

# --- Configuration for File Uploads ---
# Define the folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed image extensions for security
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

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
# IMPORTANT: Ensure your Ollama server is running and the 'llava' model is downloaded.
# You can download it using: ollama run llava
try:
    # Initialize the OllamaLLM model.
    # 'llava' is chosen here because the example includes image handling.
    # If you only need text, you can use models like 'llama3', 'mistral', etc.
    model = ChatOllama(model="qwen2.5vl:3b")
    logging.info("ChatOllama model initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize OllamaLLM: {e}")
    model = None # Set model to None if initialization fails

# --- Chat History Management (In-memory, resets on server restart) ---
# This will store the conversation history as a list of dictionaries,
# suitable for Langchain's message format.
# Example: [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi there!'}]
chat_history = []
logging.info("Chat history initialized.")


# --- Flask Routes ---

# Define a route for the home page (dashboard.html)
@app.route('/')
@app.route('/dashboard.html')
def dashboard():
    """
    Renders the dashboard.html template.
    """
    return render_template('dashboard.html')

# Define a route for the flowchart page (Test Case Generation)
@app.route('/flowchart.html')
def flowchart():
    """
    Renders the flowchart.html template.
    """
    return render_template('flowchart.html')

# Define a route for the chatbot page (QA Chatbot)
@app.route('/chatbot.html')
def chatbot():
    """
    Renders the chatbot.html template.
    """
    # When loading the chatbot page, clear the history for a fresh start
    global chat_history
    chat_history = []
    logging.info("Chatbot page loaded, chat history cleared.")
    return render_template('chatbot.html')

# Define a route for the refinement page (Test Refinement)
@app.route('/refinement.html')
def refinement():
    """
    Renders the refinement.html template.
    """
    return render_template('refinement.html')

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

    messages_for_model = []
    current_user_message_parts = []

    if user_input:
        current_user_message_parts.append({"text": user_input})

    if image_file and allowed_file(image_file.filename):
        try:
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(filepath)
            logging.debug(f"Image saved to {filepath}")
            # Ollama expects image paths for multimodal models
            current_user_message_parts.append({"image": filepath})
        except Exception as e:
            logging.error(f"Error saving image: {e}")
            return jsonify({"response": f"Error saving image: {str(e)}"}), 500
    
    if not user_input and not image_file:
        return jsonify({"response": "Please enter a message or upload an image."}), 400

    # Append current user message to global chat history
    chat_history.append({'role': 'user', 'content': current_user_message_parts})

    # Prepare messages for the model, including previous history
    # Langchain's OllamaLLM expects a list of messages.
    # The 'content' for messages can be a string or a list of dicts for multimodal.
    # We convert our internal chat_history format to the model's expected format.
    langchain_messages = []
    for msg in chat_history:
        if isinstance(msg['content'], list): # Multimodal content
            langchain_content = []
            for part in msg['content']:
                if 'text' in part:
                    langchain_content.append({"type": "text", "text": part['text']})
                if 'image' in part:
                    # For OllamaLLM, image paths are directly supported
                    langchain_content.append({"type": "image_url", "image_url": {"url": f"{os.path.abspath(part['image'])}"}} )
        else: # Text-only content
            langchain_content = msg['content']
        
        langchain_messages.append({'role': msg['role'], 'content': langchain_content})

    logging.debug(f"Messages sent to model: {langchain_messages}")

    # try:
    #     # Get response from the model
    #     response = model.invoke(langchain_messages)
    #     logging.debug(f"Model raw response: {response}")

    #     # Append bot response to global chat history
    #     chat_history.append({'role': 'assistant', 'content': response})
    #     logging.debug(f"Updated chat history: {chat_history}")

    #     # Send response to the front-end
    #     return jsonify({"response": response})
    # except Exception as e:
    #     logging.error(f"Error invoking Ollama model: {e}")
    #     return jsonify({"response": f"Error: {str(e)}. Ensure Ollama server is running and model ('llava') is downloaded."}), 500

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

# This block ensures that the Flask development server runs only when
# the script is executed directly (e.g., `python app.py` or `flask run`).
if __name__ == '__main__':
    app.run(debug=True)