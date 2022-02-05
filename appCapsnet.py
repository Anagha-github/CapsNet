import streamlit as st
import subprocess
import sys
from image_capt import image_capture
from prediction import predict_image
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

#########################
# arr = np.random.normal(1, 1, size=100)
# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)

# st.pyplot(fig)

####################

up_file = st.file_uploader("upload")
if up_file is not None:
    image = Image.open(up_file)
    image = np.array(image)
    # image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    st.image(image)

    # img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    # image = cv2.imread(img_file_buffer)
    # # image = Image.open(img_file_buffer)
    

    preprocessed_digits = image_capture(image)
    y_pred_value_array = predict_image(preprocessed_digits)
    i=0
    # fig1 = plt.figure(figsize = (3,3))
    for digit in preprocessed_digits:
        
        # plt.subplot(len(preprocessed_digits), 1, (i+1)).set_title(str(y_pred_value_array[i]), fontsize = 10)
        # plt.imshow(digit, cmap = 'gray')
        # plt.axis('off')
        # plt.(str(y_pred_value_array[i]), fontsize = 10)

        # axs = axs.ravel()

        # axs[i].imshow(digit,cmap = 'gray')
        # axs[i].set_title(str(y_pred_value_array[i]))
        st.image(digit)   
        st.write("Predicted value = ")
        st.write(str(y_pred_value_array[i]))
        i=i+1
    #     plt.subplots_adjust(wspace=2, hspace=2)
    # st.pyplot(fig1)
    # st.pyplot(fig)
        
   
    # st.write(y_pred_value_array)

    # subprocess.run([f"{sys.executable}", "RandomHandWriting.py"])
