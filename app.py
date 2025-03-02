from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# ✅ Load OpenAI API Key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")  # Use correct env variable name

if not openai_api_key:
    raise ValueError("Missing OpenAI API Key! Add it to .env")

client = openai.OpenAI(api_key=openai_api_key)

def encode_image(image):
    """Converts an image to Base64 (required for OpenAI Vision API)."""
    img_io = io.BytesIO()
    image.save(img_io, format="PNG")  # Convert image to PNG format
    img_io.seek(0)
    return base64.b64encode(img_io.read()).decode("utf-8")

def rate_food(image):
    """Sends the image to OpenAI's GPT-4 Turbo Vision API and gets a food rating."""
    image_data = encode_image(image)

    response = client.chat.completions.create(
        model="gpt-4-turbo",  # ✅ Updated model name
        messages=[
            {"role": "system", "content": "You are a professional food critic. Analyze the food in the image and rate it from 1 to 10 based on its presentation. Suggest one improvement."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is an image of a dish. Please rate it from 1 to 10 and provide a suggestion to improve its presentation."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]
            }
        ],
        max_tokens=100,
        temperature=0.7
    )

    return response.choices[0].message.content  # Extract the AI response

@app.route("/upload", methods=["POST"])
def upload_image():
    """Handles image upload and sends it to OpenAI for rating."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    
    try:
        image = Image.open(file)  # Open the image using PIL
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    rating = rate_food(image)  # Send image to OpenAI API

    return jsonify({"rating": rating})  # Return the AI-generated rating

if __name__ == "__main__":
    app.run(debug=True, port=5000)
