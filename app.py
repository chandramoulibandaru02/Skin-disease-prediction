import os
import pickle
from flask import Flask, request, render_template_string
from PIL import Image
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# --- Load the Model ---
# Ensure your 'skin_model.pkl' is in the same directory as this script
MODEL_PATH = 'skin_model.pkl'
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'. Please make sure the file exists.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Define Class Labels ---
# IMPORTANT: Update this list to match the classes your model predicts, in the correct order.
DISEASE_CLASSES = [
    "Eczema", "Melanoma", "Atopic Dermatitis", "Basal Cell Carcinoma (BCC)",
    "Melanocytic Nevi (NV)", "Benign Keratosis-like Lesions (BKL)",
    "Psoriasis & Lichen Planus", "Seborrheic Keratoses & other Benign Tumors",
    "Tinea, Ringworm, Candidiasis & other Fungal Infections",
    "Warts, Molluscum & other Viral Infections"
]
def preprocess_image(image_file):
    """
    Preprocesses the uploaded image to match the model's expected input format.
    """
    # Open the image file
    img = Image.open(image_file.stream).convert('RGB')
    
    # --- IMPORTANT: Resize to the input size your model expects ---
    # Common sizes are (224, 224), (160, 160), etc.
    img = img.resize((224, 224))
    
    # Convert image to a NumPy array
    img_array = np.array(img)
    
    # Normalize pixel values (if your model expects this)
    img_array = img_array / 255.0
    
    # Add a batch dimension (e.g., from (224, 224, 3) to (1, 224, 224, 3))
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


@app.route('/', methods=['GET', 'POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error="Model is not loaded. Please check server logs.")

    if request.method == 'POST':
        # Check if an image was uploaded
        if 'image' not in request.files:
            return render_template_string(HTML_TEMPLATE, error="No image file provided.")
        
        file = request.files['image']
        
        # Check if the filename is empty
        if file.filename == '':
            return render_template_string(HTML_TEMPLATE, error="No image selected.")

        try:
            # Preprocess the image
            processed_image = preprocess_image(file)

            # Make a prediction
            preds = model.predict(processed_image)
            
            # Get the predicted class index
            predicted_index = np.argmax(preds, axis=1)[0]
            
            # Get the corresponding class label
            predicted_class = DISEASE_CLASSES[predicted_index]

            return render_template_string(HTML_TEMPLATE, prediction=predicted_class)

        except Exception as e:
            # Handle potential errors during processing or prediction
            print(f"An error occurred: {e}")
            return render_template_string(HTML_TEMPLATE, error="Failed to process image or make prediction.")

    # For a GET request, just display the upload form
    return render_template_string(HTML_TEMPLATE)


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)