import cv2

def preprocess_image(image):
    # Resize to 28x28
    resized = cv2.resize(image, (28, 28))

    # Increase contrast
    alpha = 1.5  # Contrast control
    beta = 0     # Brightness control
    contrast = cv2.convertScaleAbs(resized, alpha=alpha, beta=beta)

    # Binary thresholding
    _, thresholded = cv2.threshold(contrast, 128, 255, cv2.THRESH_BINARY)

    # Normalize and return
    return thresholded
