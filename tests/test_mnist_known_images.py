import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import load_trained_model
from preprocessing.image_processor import AdaptivePreprocessor

def run_test():
    print("Loading MNIST test dataset...")
    (_, _), (x_test, y_test) = mnist.load_data()
    
    print("Loading trained MNIST model...")
    model = load_trained_model("mnist")
    if model is None:
        print("Error: Could not load MNIST model. Please run training script first.")
        return
    
    # Test on a known set of images (e.g. first 10, which usually cover various digits)
    # We will pick one image for each digit (0-9)
    print("\nRunning validation on known MNIST images (0-9)...\n")
    
    tested_digits = set()
    correct = 0
    total = 0
    
    preprocessor = AdaptivePreprocessor(target_size=(28, 28))
    
    for i in range(len(x_test)):
        digit = y_test[i]
        if digit not in tested_digits:
            tested_digits.add(digit)
            img = x_test[i]
            
            # The model expects images to be normalized between 0 and 1, with shape (1, 28, 28, 1)
            # We'll pass the raw image through our preprocessing pipeline to verify it doesn't break perfect images
            processed_model, _ = preprocessor.preprocess(img, for_model=True)
            
            # Ensure correct shape for model prediction
            if processed_model.ndim == 2:
                processed_model = processed_model.reshape(1, 28, 28, 1)
            
            # Model prediction
            prediction_probs = model.predict(processed_model, verbose=0)
            predicted_digit = np.argmax(prediction_probs[0])
            confidence = np.max(prediction_probs[0])
            
            is_correct = predicted_digit == digit
            if is_correct:
                correct += 1
            total += 1
            
            print(f"True Digit: {digit} | Predicted: {predicted_digit} | Confidence: {confidence*100:.2f}% | Correct: {is_correct}")
            
            if len(tested_digits) == 10:
                break
                
    print(f"\nValidation Result: {correct}/{total} correct.")
    if correct == total:
        print("SUCCESS! The model and preprocessing pipeline are correctly recognizing standard MNIST images.")
    else:
        print("WARNING! The model failed on some standard MNIST images. Please verify the preprocessing pipeline or retraining.")

if __name__ == "__main__":
    run_test()
