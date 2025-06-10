from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
model = load_model('tb_detection_model.h5')

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']

    if file.filename == '':
        return 'No selected file', 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        # Correct preprocessing for CNN expecting (128, 128, 1)
        img = Image.open(image_path).convert('L')  # Grayscale
        img = img.resize((128, 128))               # Resize to 128x128
        img = np.array(img).astype('float32') / 255.0
        img = img.reshape(1, 128, 128, 1)           # Add batch and channel dims

        prediction = model.predict(img)[0][0]
        result = "Tuberculosis Detected" if prediction > 0.5 else "No Tuberculosis"

        return render_template('result.html', prediction=result, image_path=image_path)

    return 'Invalid file type', 400

if __name__ == '__main__':
    app.run(debug=True)
