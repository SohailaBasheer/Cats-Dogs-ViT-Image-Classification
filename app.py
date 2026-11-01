
import streamlit as st
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

st.title("🐱 Cats vs Dogs Classifier")
st.write("Upload an image to identify whether it is a Cat or Dog.")

MODEL_NAME = "Sohaila16/cats-dogs-vit-classifier"

@st.cache_resource
def load_model():
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model

processor, model = load_model()

uploaded_file = st.file_uploader(
    "Upload a Cat or Dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    if st.button("Predict"):
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = outputs.logits.argmax(-1).item()
        label = model.config.id2label[prediction]

        st.success(f"Prediction: {label}")
