from flask import Flask, request, jsonify
import easyocr
import cv2
import numpy as np
import os
import ocr  # Assuming you have an ocr.py module for processing the image in 'deteksirangka'
from werkzeug.utils import secure_filename
from detect_refactor import detect_vin  # Import your VIN detection function

# Initialize Flask app
app = Flask(__name__)

# Initialize EasyOCR reader for 'deteksinopol'
reader = easyocr.Reader(['en'], gpu=True)

# Configure upload folder and file size limits
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit for all endpoints

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Function to process the OCR results for 'deteksinopol'
def process_ocr_results(results, img_shape):
    nopol_parts = []
    
    # Define a threshold to filter out the bottom line (e.g., date) based on Y-coordinate
    img_height = img_shape[0]
    first_line_threshold_y = img_height * 0.5  # Everything above 50% of the image height is considered "top line"
    
    for (bbox, text, prob) in results:
        # Get the Y-coordinates of the top-left and bottom-left corners
        top_left_y = bbox[0][1]
        bottom_left_y = bbox[3][1]
        
        # Calculate the average Y-coordinate for the bounding box
        bbox_y_avg = (top_left_y + bottom_left_y) / 2
        
        # Filter out text that's in the lower part of the image (below the threshold)
        if bbox_y_avg < first_line_threshold_y:
            nopol_parts.append(text)
    
    # Join the license plate parts
    nopol = " ".join(nopol_parts)
    return nopol


# Helper function to convert image file to OpenCV format
def read_image_from_request(image):
    # Convert the image file to a format suitable for OpenCV (numpy array)
    file_bytes = np.frombuffer(image.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


# Route for 'deteksinopol'
@app.route('/deteksinopol', methods=['POST'])
def deteksi_nopol():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    img = read_image_from_request(file)

    # Perform OCR on the image using EasyOCR
    results = reader.readtext(img)

    # Process results to get the formatted output (only top line)
    nopol = process_ocr_results(results, img.shape)

    # Return the result as JSON
    return jsonify({"nopol": nopol})


# Route for 'deteksirangka'
@app.route('/deteksirangka', methods=['POST'])
def deteksi_rangka():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img = read_image_from_request(file)

    # Save the image temporarily for OCR processing in deteksi rangka
    temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
    cv2.imwrite(temp_image_path, img)

    # Run OCR function from ocr.py for deteksi rangka
    result = ocr.process_image(temp_image_path)

    # Delete the temporary image after processing
    os.remove(temp_image_path)

    return jsonify({"result": result}), 200


# Route for 'predict-vin'
@app.route('/predict-vin', methods=['POST'])
def predict_vin():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Call the VIN detection function
        vin_sequence = detect_vin(source=file_path)

        return jsonify({'predicted_vin': vin_sequence}), 200


# Main function to run the app
if __name__ == '__main__':
    # Ensure upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    app.run(host='0.0.0.0', port=4000, debug=True)
