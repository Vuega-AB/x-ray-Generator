import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Load the generator model
G = tf.keras.models.load_model('generator2.h5')  # Ensure 'generator_model' is the correct model path

# Streamlit app title
st.title('Chest X-Ray Image Generator')

# Select the number of images to generate
num_images = st.slider('Select the number of images to generate', 1, 64, 16)

# Choose the label (Normal or Pneumonia)
label = st.selectbox('Select the label for generated images', ['Normal', 'Pneumonia'])
label_encoding = 0 if label == 'Normal' else 1

if st.button('Generate Images'):
    noise = tf.random.uniform(shape=(num_images, 100), minval=-1, maxval=1)
    labels = tf.keras.utils.to_categorical([label_encoding] * num_images, num_classes=2)
    
    generated_images = G.predict([noise, labels])
    
    # Create a figure for the generated images
    fig = plt.figure(figsize=(12, 12))
    
    for i in range(num_images):
        plt.subplot(8, 8, i + 1)  # Create a grid of subplots
        img = (generated_images[i] * 255).astype(np.uint8).squeeze()
        plt.imshow(img, cmap='gray')
        plt.title('Normal' if label_encoding == 0 else 'Pneumonia')
        plt.axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
