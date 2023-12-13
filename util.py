import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, class_names):
    # Convert the image to grayscale
    image = image.convert('L')
    image = image.resize((224, 224))
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 1), dtype=np.float32)  # Change the shape to (1, 224, 224, 1)
    data[0] = normalized_image_array[:, :, np.newaxis]

    # Initialize prediction variable
    prediction = None

    # Print information for debugging
    print(f"Data shape before prediction: {data.shape}")

    try:
        prediction = model.predict(data)
        print(f"Raw Prediction: {prediction}")
    except Exception as e:
        print(f"Error during prediction: {e}")

    if prediction is not None and len(prediction) > 0:
        # Print information for debugging
        print(f"Prediction: {prediction}")

        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        return class_name, confidence_score

    # Return default values if prediction is None or empty
    return "Unknown", 0.0
