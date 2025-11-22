import streamlit as st
from fastai.vision.all import *
import streamlit as st

st.title("Number Identifier")
st.text("Built By Joshua")

def number_identifier(file_name):
    file_parts = str(file_name).split("/")
    folder_name = file_parts[-2]
    return folder_name[-1]

number_classification_model = load_learner("single_digit_model.pkl")

def predict(image):
    img = PILImage.creat(image)
    pred_class, pred_idx, outputs = number_classification_model.predict(img)
    return pred_class

uploaded_file = st.file_uploader("Choose an image to upload...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    prediction = predict(uploaded_file)
    st.subheader(f"Predicted number: {prediction}")

st.text("Built with Streamlit and FastAI.")