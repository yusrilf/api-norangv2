from flask import Flask, request, jsonify
import os
from detect_refactor import detect_vin  # Import your VIN detection function
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload to 16 MB

# Route to handle image upload and VIN prediction
@app.route('/predict-vin', methods=['POST'])
def predict_vin():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Call the VIN detection function
        vin_sequence = detect_vin(source=file_path)

        return jsonify({'predicted_vin': vin_sequence}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
