import streamlit as st
import subprocess
import sys
from image_capt import image_capture
from prediction import predict_image
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

up_file = st.file_uploader("upload")
if up_file is not None:
    image = Image.open(up_file)
    image = np.array(image)
    st.image(image)

    preprocessed_digits = image_capture(image)
    y_pred_value_array = predict_image(preprocessed_digits)
    i=0
    for digit in preprocessed_digits:
        
        st.image(digit)   
        st.write("Predicted value = ")
        st.write(str(y_pred_value_array[i]))
        i=i+1
    
