# Test Case Generation

This project is an ongoing application with three main features designed to assist with Software Quality Assurance (SQA) tasks and for fullfiling my bachelor degree Thesis.
## How to Run

To run this project, you'll need to install the following libraries:

1.  For test case generation:
    `pip install easyocr opencv-python-headless requests pandas openpyxl Pillow numpy google-generativeai inference-sdk python-dotenv --quiet`
    (Alternatively, if `inference-sdk` causes issues, you can use:
    `pip install easyocr opencv-python-headless requests pandas openpyxl Pillow numpy google-generativeai --quiet`)
2.  For SQA Chatbot, mandatory to connect and run the LLM (with Ollama):
    `pip install Flask langchain-ollama python-dotenv`
    see more on : https://ollama.com/library/gemma3

## Features

This application provides the following functionalities:

1.  Generate Test Cases using Flowcharts (LLM-powered)
    > OCR with EasyOCR: Processes images with techniques like motion blur and resizing for accurate text extraction. Specific settings, such as language (e.g., Indonesian), are applied to improve reading probability.
    
    > LLM Integration: Utilizes models like Llama (Llama 3.2, Llava, Minicpm) or Gemini, with configurable temperature settings.
    
    > Image Segmentation: Using Roboflow, with adjustable confidence and overlap settings.

2.  Chatbot AI for QA Tasks
    * A specialized SQA chatbot powered by LLMs with dedicated prompt engineering.

3.  Refinement of Test Cases According to KBBI Guidelines
    * Utilizes a BaseLLM with prompt engineering to define and apply standard KBBI (Kamus Besar Bahasa Indonesia) template guidelines. This includes removing redundancies, combining similar cases, and enhancing overall efficiency.
    * Provides Test Case Review, offering a point summary, identified strengths, and suggestions for improvement.
