import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import pandas as pd

# ----------------- CONFIGURATION -----------------
MODEL_PATH = "teeth_classifier_model.keras"
CLASS_NAMES = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]
IMAGE_SIZE = (256, 256)


# ----------------- LOAD MODEL -----------------
@st.cache_resource
def load_keras_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


model = load_keras_model()

# ----------------- SET PAGE CONFIG -----------------
st.set_page_config(
    page_title="Dental Disease Classifier", layout="centered", page_icon="🦷"
)

# ----------------- HEADER -----------------
st.title("🦷 AI Teeth Disease Classifier")
st.markdown("Upload a dental image below to get a prediction.")

st.markdown("---")

# ----------------- FILE UPLOAD -----------------
uploaded_file = st.file_uploader(
    "📤 Upload a tooth image (JPG, PNG):",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear, high-resolution image of the tooth area.",
)


# ----------------- PREPROCESSING & PREDICTION -----------------
def preprocess_image_and_predict(image_to_process, target_size=IMAGE_SIZE):
    img = image_to_process.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array_expanded)
    prediction = model.predict(processed_img)
    return prediction


# ----------------- SHOW IMAGE & PREDICTION -----------------
if model is None:
    st.stop()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    if st.button("🔍 Classify Image"):
        with st.spinner("Analyzing the image..."):
            prediction = preprocess_image_and_predict(image)
            predicted_index = np.argmax(prediction)
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = np.max(prediction)

            st.success(f"**✅ Predicted Class:** `{predicted_class}`")
            st.info(f"**📊 Confidence:** `{confidence:.2%}`")

            # Show all class probabilities
            prob_dict = {
                CLASS_NAMES[i]: float(prediction[0][i]) for i in range(len(CLASS_NAMES))
            }
            prob_df = pd.DataFrame.from_dict(
                prob_dict, orient="index", columns=["Probability"]
            )
            st.bar_chart(prob_df)

            st.markdown("---")

# ----------------- FOOTER -----------------
with st.expander("ℹ️ About this App"):
    st.write("""
        - This app uses a deep learning model trained on 7 dental conditions.
        
        - Input size: 256x256 RGB images

        - Output: One of 7 categories with confidence scores.
        
        - Built with 💙 using **TensorFlow** and **Streamlit**
    """)
