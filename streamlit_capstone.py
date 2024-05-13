import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your pre-trained CNN model
model = tf.keras.models.load_model('model_classification_original_7000_dataset.keras')

# Define class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']  # Adjust based on your classes

# Function to preprocess image
def preprocess_image(image):
    # Resize image to match model input size
    img = image.resize((256, 256))
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values
    return img

# Streamlit UI
st.title('CNN Image Classifier')

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Classify'):
        # Preprocess the image
        processed_img = preprocess_image(image)
        # Reshape for model input (add batch dimension)
        processed_img = np.expand_dims(processed_img, axis=0)
        
        # Perform inference
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions)
        
        # Display results
        st.write(f'Predicted class: {class_labels[predicted_class]}')
        st.write(f'Confidence: {predictions[0][predicted_class]*100:.2f}%')