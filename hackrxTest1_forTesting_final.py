# To explore mobilenetv2
# To explore VGG16
# import os#all below imported packeages are important
from kmeans_for_colourClustering import find_number_of_colors
from hackrx_TextMeasure import rate_text_amount 
from keras.models import load_model
import cv2
from PIL import Image
import numpy as np
import streamlit as st
# import pandas as pd # pip install pandas
# from matplotlib import pyplot as plt # pip install matplotlib
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#Used to avoid printing of Warnings
st.title("Deep Learning based Visual Testing Tool for Website Design Validation Developed by PixelProphets")
# get the path/directory
folder_dir = "C://aditi//competitions//hackerx4//test"

# streamlite commands

st.set_option('deprecation.showfileUploaderEncoding', False)
#st.title('Website Design Assessment Tool using Deep Learning')
st.text(
    'Upload the Image from the listed category.\n[good design or bad design]')

saved_model1 = load_model('C://aditi//competitions//hackerx4//best_model_mobilenet1.hdf5')
saved_model2 = load_model('C://aditi//competitions//hackerx4//best_model_mobilenet_forContrast.hdf5')
saved_model3 = load_model('C://aditi//competitions//hackerx4//best_model_mobilenet_forLayout.hdf5')
saved_model4 = load_model("C://aditi/competitions/hackerx4/best_model_mobilenet_forClutter.hdf5")

uploaded_file = st.file_uploader("Choose an image...", type=[
                                 "jpg", "png", "tiff", "bmp"])
if uploaded_file is not None:
    imge = Image.open(uploaded_file)
    st.image(imge, caption='Uploaded Image')

    if st.button('PREDICT'):
        Categories1 = ['bad_website_design', 'good_website_design']
        Categories2 = ['bad_website_contrast', 'good_website_contrast']
        Categories3 = ['bad_website_layout', 'good_website_layout']
        Categories4 = ['cluttered_website', 'non_cluttered_website']

        st.write('Result.....')
        flat_data = []
        imge = np.array(imge)
        print(imge.shape)
        # The INTER_NEAREST method uses the nearest neighbor concept for interpolation. This is one of the simplest methods, using only one neighboring pixel from the image for interpolation.
        img1 = cv2.resize(imge, (224, 224), interpolation=cv2.INTER_NEAREST)
        norm_image = cv2.normalize(
            img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if norm_image.shape[2] == 4:
            #slice off the alpha channel
            norm_image = norm_image[:, :, :3]

        print(norm_image.shape) 

        y = np.expand_dims(norm_image, axis=0)
        print(y.shape)

        y_out1 = saved_model1.predict(y)
        y_out1 = np.round(y_out1)

        y_out2 = saved_model2.predict(y)
        y_out2 = np.round(y_out2)

        y_out3 = saved_model3.predict(y)
        y_out3 = np.round(y_out3)

        y_out6 = saved_model4.predict(y)
        y_out6 = np.round(y_out6)

        y_out1 = Categories1[y_out1.argmax()]
        y_out2 = Categories2[y_out2.argmax()]
        y_out3 = Categories3[y_out3.argmax()]
        y_out6 = Categories4[y_out6.argmax()]

        totalUniqueColors = find_number_of_colors(uploaded_file) 
        y_out4="ADEQUATE NUMBER OF COLORS USED IN WEBSITE DESIGN"
        if totalUniqueColors>6:
            y_out4="TOO MANY COLORS USED IN WEBSITE DESIGN"

        y_out5 = rate_text_amount(uploaded_file)

        #st.title(f' PREDICTED OUTPUT: {y_out1}')
        st.title(f' PREDICTED OUTPUT: {y_out2}')
        st.title(f' PREDICTED OUTPUT: {y_out3}')
        st.title(f' PREDICTED OUTPUT: {y_out4}')
        st.title(f' PREDICTED OUTPUT: {y_out5}')
        st.title(f' PREDICTED OUTPUT: {y_out6}')

        # q = saved_model.predict_proba(y)
        # for index, item in enumerate(Categories):
        #   st.write(f'{item} : {q[0][index]*100}%')

st.text("")
st.text('Made by PixelProphets')
