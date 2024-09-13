import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization

# Step 1: Define the model architecture
def create_model():
    base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))

    # Freeze the bottom layers of ResNet50 model and unfreeze the top layers for fine-tuning
    for layer in base_model.layers[:143]:
        layer.trainable = False
    for layer in base_model.layers[143:]:
        layer.trainable = True

    # Adding custom layers on top of ResNet50 base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Step 2: Load the model and weights
saved_weights_path = 'brain_tumor_model.h5'  # Update with your actual path
model = create_model()
model.load_weights(saved_weights_path)

# Step 3: Preprocess the input image and make prediction
IMG_SIZE = (224, 224)
class_labels = ['Brain Tumor', 'Healthy']  # Define your class labels

def predict_image(img_path, model):
    # Load the image and resize
    img = load_img(img_path, target_size=IMG_SIZE)
    
    # Convert image to array
    img_array = img_to_array(img)
    
    # Preprocess the image for ResNet50
    img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))
    
    # Make prediction
    prediction = model.predict(img_preprocessed)
    
    # Determine class label
    predicted_label = class_labels[int(prediction[0] > 0.5)]  # Binary classification (0: Healthy, 1: Brain Tumor)

    return predicted_label

# Initialize Flask app
app = Flask(__name__)

# Ensure the 'uploads' folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the main route to handle image upload
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded.'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected.'})

        # Save the uploaded image to a temporary path
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        # Make prediction on the uploaded image
        predicted_label = predict_image(img_path, model)

        # Clean up: remove the image after prediction
        os.remove(img_path)

        # Return the prediction as JSON
        return jsonify({'prediction': f'The model predicts: {predicted_label}'})
    
    return render_template('upload.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
