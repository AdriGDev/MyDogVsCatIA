import streamlit as st
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing import image
import numpy as np
from PIL import Image


model = load_model("artemisa.h5")

# function to process any image to be recognizable by my IA
def preprocess_image(img, target_size=(100, 100)):
    img = img.resize(target_size)
    img = img.convert("L")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


st.title("Dogs vs Cats")
uploaded_file = st.file_uploader("Upload any image of your Cat/Dog", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Image", use_container_width=True)
    # Send the processed image to my IA
    prediction = "**Its a Dog!**" if model.predict(preprocess_image(Image.open(uploaded_file)))[0]>0.5 else "**Its a Cat!**"
    st.write(prediction)