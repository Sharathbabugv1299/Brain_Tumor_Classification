from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
import os

# Step 1: Define the model architecture (same as your original model)
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


app = Flask(__name__)

# Load the model once when the app starts
model = create_model()
model.load_weights('best_model_weights.h5')  # Update with actual path

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the uploaded file temporarily
        img_path = os.path.join('temp_images', file.filename)
        file.save(img_path)
        
        # Predict the class of the image
        predicted_label = predict_image(img_path, model)
        
        # Clean up by removing the temporary file
        os.remove(img_path)
        
        return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
