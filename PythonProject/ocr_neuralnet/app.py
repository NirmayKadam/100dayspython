from flask import Flask, render_template, request
import cv2
import numpy as np
import pickle
from utils.preprocess import preprocess_image
from model.neural_network import NeuralNetwork

# Initialize Flask app
app = Flask(__name__)

# Load trained model weights
import os

weights_path = os.path.join(os.getcwd(), 'model', 'nn_weights.pkl')
print("Looking for the weights at:", weights_path)

with open(weights_path, 'rb') as f:
    weights_biases = pickle.load(f)



# Initialize the neural network
nn = NeuralNetwork(input_size=784, hidden_layers=(128, 64), output_size=10)
nn.set_weights(weights_biases)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('prediction.html', prediction='No image file uploaded.')

    file = request.files['image']
    if file.filename == '':
        return render_template('prediction.html', prediction='No image selected.')

    # Read and preprocess the image
    image_bytes = file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    processed_image = preprocess_image(image)

    # Flatten and normalize for NN input
    flat_image = processed_image.flatten() / 255.0

    # Predict using the neural network
    output = nn.predict(flat_image.reshape(1, -1))
    prediction = np.argmax(output)

    return render_template('prediction.html', prediction=f'Predicted digit: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
