from flask import Flask, render_template, request, redirect, url_for
import cv2
import pytesseract
import numpy as np
import base64


import pytesseract

# Optional: Explicitly set the path in your code
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)


def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=50)

    # Resize image (scale up by 2x for better OCR accuracy)
    resized = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(resized, None, 30, 7, 21)

    # Apply binary thresholding
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    return binary


def read_text(image):
    img_bytes = image.read()
    img_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    processed_image = preprocess_image(img)
    text = pytesseract.image_to_string(processed_image)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(img_encoded).decode('utf-8')
    return text, img_str


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def get_predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('predict.html', text='No image file selected', img_str=None)
        file = request.files['image']
        if file.filename == '':
            return render_template('predict.html', text='No image file selected', img_str=None)

        image_text, img_str = read_text(file)
        return render_template('predict.html', text=image_text, img_str=img_str)
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
