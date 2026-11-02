
import streamlit as st
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐱",
    layout="centered"
)

# ---------------- CUSTOM BACKGROUND ----------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        -45deg,
        #0f172a,
        #312e81,
        #581c87,
        #164e63
    );
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
    color: white;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Floating bubbles */
.stApp::before {
    content: "";
    position: fixed;
    width: 250px;
    height: 250px;
    top: 10%;
    left: 5%;
    background: rgba(255,255,255,0.10);
    border-radius: 50%;
    filter: blur(3px);
    animation: float 8s ease-in-out infinite;
    pointer-events: none;
}

.stApp::after {
    content: "";
    position: fixed;
    width: 200px;
    height: 200px;
    bottom: 8%;
    right: 5%;
    background: rgba(236,72,153,0.20);
    border-radius: 50%;
    filter: blur(5px);
    animation: float 10s ease-in-out infinite reverse;
    pointer-events: none;
}

@keyframes float {
    0%, 100% {transform: translateY(0px);}
    50% {transform: translateY(-40px);}
}

/* Main content */
.block-container {
    max-width: 800px;
    padding-top: 3rem;
    padding-bottom: 3rem;
    position: relative;
    z-index: 1;
}

/* Headings */
h1 {
    color: white !important;
    text-align: center;
    font-size: 42px !important;
    font-weight: 800 !important;
    text-shadow: 2px 3px 12px rgba(0,0,0,0.4);
}

p, label, .stMarkdown {
    color: white !important;
}

/* Upload box */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.13);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.3);
}

/* Uploaded image */
[data-testid="stImage"] img {
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.35);
}

/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #ec4899, #8b5cf6);
    color: white;
    border: none;
    border-radius: 30px;
    padding: 12px;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.04);
    box-shadow: 0 5px 25px rgba(236,72,153,0.5);
    color: white;
    border: none;
}

/* Success message */
[data-testid="stAlert"] {
    background: rgba(34,197,94,0.18);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.3);
}

/* Remove Streamlit top decoration */
[data-testid="stHeader"] {
    background: transparent;
}

</style>
""", unsafe_allow_html=True)


# ---------------- APP TITLE ----------------

st.title("🐱 Cats vs Dogs Classifier")

st.markdown(
    "<h4 style='text-align:center; color:#e0e7ff;'>"
    "🐾 Smart AI Image Recognition 🐾"
    "</h4>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>"
    "Upload an image and let Artificial Intelligence identify your pet!"
    "</p>",
    unsafe_allow_html=True
)

st.write("")

# ---------------- LOAD MODEL ----------------

MODEL_NAME = "Sohaila16/cats-dogs-vit-classifier"

@st.cache_resource
def load_model():
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model

processor, model = load_model()

# ---------------- IMAGE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "📸 Upload a Cat or Dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Your Uploaded Pet Image",
        use_container_width=True
    )

    if st.button("✨ Predict Image"):

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = outputs.logits.argmax(-1).item()
        label = model.config.id2label[prediction]

        st.success(f"🐾 Prediction: {label}")

        st.balloons()

else:
    st.markdown(
        "<h3 style='text-align:center; color:#fbcfe8;'>"
        "🐶 Waiting for your lovely pet image! 🐱"
        "</h3>",
        unsafe_allow_html=True
    )

# ---------------- FOOTER ----------------

st.markdown("""
<hr style="border:1px solid rgba(255,255,255,0.25);">

<p style="text-align:center; color:#ddd6fe;">
Made with ❤️ using Deep Learning, Vision Transformer & Streamlit
</p>
""", unsafe_allow_html=True)
