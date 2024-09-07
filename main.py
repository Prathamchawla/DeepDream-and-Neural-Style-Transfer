import streamlit as st
from PIL import Image
import torch
import DeepDream  # Assuming the DeepDream functions are in DeepDream.py
import NST  # Assuming the Neural Style Transfer functions are in NST.py

# Title of the web app
st.title("DeepDream and Neural Style Transfer Web App")

# Main page options
option = st.selectbox('Choose a Model', ['Select', 'DeepDream', 'Neural Style Transfer'])

if option == 'DeepDream':
    st.header("DeepDream")
    
    # Upload image for DeepDream
    uploaded_image = st.file_uploader("Upload an Image for DeepDream", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image is not None:
        # Open the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # DeepDream Parameters
        st.sidebar.header('DeepDream Parameters')
        iterations = st.sidebar.slider('Iterations', min_value=1, max_value=50, value=10)
        learning_rate = st.sidebar.slider('Learning Rate (LR)', min_value=0.01, max_value=0.5, value=0.1)
        num_octaves = st.sidebar.slider('Number of Octaves', min_value=1, max_value=10, value=3)
        octave_scale = st.sidebar.slider('Octave Scale', min_value=1.1, max_value=2.0, value=1.5)
        
        # Apply DeepDream
        if st.button('Apply DeepDream'):
            st.write("Processing...")
            
            # Call your DeepDream function here
            try:
                layer = list(DeepDream.vgg.features.modules())[10]  # Adjust the layer index as needed
                output_image = DeepDream.deepdream_octaves(image, iterations, learning_rate, num_octaves, octave_scale)
                
                # Display the output image
                st.image(output_image, caption='DeepDream Output Image', use_column_width=True)
                st.success("DeepDream processing complete!")
            except Exception as e:
                st.error(f"An error occurred while applying DeepDream: {e}")

elif option == 'Neural Style Transfer':
    st.header("Neural Style Transfer")
    
    # Upload content and style images for Neural Style Transfer
    uploaded_content_image = st.file_uploader("Upload Content Image", type=['jpg', 'jpeg', 'png'], key='content')
    uploaded_style_image = st.file_uploader("Upload Style Image", type=['jpg', 'jpeg', 'png'], key='style')
    
    if uploaded_content_image is not None and uploaded_style_image is not None:
        # Open the uploaded images
        content_image = Image.open(uploaded_content_image)
        style_image = Image.open(uploaded_style_image)
        
        st.image(content_image, caption='Uploaded Content Image', use_column_width=True)
        st.image(style_image, caption='Uploaded Style Image', use_column_width=True)
        
        # NST Parameters
        st.sidebar.header('Neural Style Transfer Parameters')
        num_steps = st.sidebar.slider('Number of Steps', min_value=100, max_value=1000, value=300)
        style_weight = st.sidebar.slider('Style Weight', min_value=10000, max_value=10000000, value=1000000)
        content_weight = st.sidebar.slider('Content Weight', min_value=1, max_value=1000, value=1)
        
        # Apply Neural Style Transfer
        if st.button('Apply Neural Style Transfer'):
            st.write("Processing...")
            
            # Call your Neural Style Transfer function here
            try:
                output_image = NST.neural_style_transfer(content_image, style_image, num_steps, style_weight, content_weight)
                
                # Display the output image
                st.image(output_image, caption='Neural Style Transfer Output Image', use_column_width=True)
                st.success("Neural Style Transfer processing complete!")
            except Exception as e:
                st.error(f"An error occurred while applying Neural Style Transfer: {e}")

# Run this script with: streamlit run main.py
