from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

# Define class labels manually
class_names = ['Bumblefoot', 'Fowlpox', 'Healthy', 'Unlabeled', 'coryza', 'crd']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'})

    try:
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        return jsonify({'prediction': predicted_class})

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({'error': 'Prediction failed'})

if __name__ == '__main__':
    print("✅ Starting Flask server...")
    app.run(debug=True)
