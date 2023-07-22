import cv2
import numpy as np
from PIL import Image
import streamlit as st
from skimage.metrics import structural_similarity
import tempfile 
import os

# Streamlit app
st.set_option('deprecation.showfileUploaderEncoding', False)
st.text('Upload the Image from the listed category.\n[good design or bad design]')
sensitivity = st.slider("Select the value of the variable:", min_value=250, max_value=40000, value=50, step=50)


uploaded_file1 = st.file_uploader("Choose the first photo", type=["jpg", "jpeg", "png"])
uploaded_file2 = st.file_uploader("Choose the second photo", type=["jpg", "jpeg", "png"])

if uploaded_file1 and uploaded_file2 is not None:
    image1 = Image.open(uploaded_file1)
    st.image(image1, caption='First Photo', use_column_width=True)
    image2 = Image.open(uploaded_file2)
    st.image(image2, caption='Second photo')

    # Save the images to temporary files
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file1, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file2:
        image1.save(temp_file1.name)
        image2.save(temp_file2.name)

        # Read the images using cv2.imread()
        before = cv2.imread(temp_file1.name)
        after = cv2.imread(temp_file2.name)

        if st.button('COMPARE'):
            st.write('Result.....')
            # Convert images to grayscale
            before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
            after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

            # Compute SSIM between the two images
            (score, diff) = structural_similarity(before_gray, after_gray, full=True)
            st.write("Image Similarity: {:.4f}%".format(score * 100))

            # The diff image contains the actual image differences between the two images
            # and is represented as a floating point data type in the range [0,1] 
            # so we must convert the array to 8-bit unsigned integers in the range
            # [0,255] before we can use it with OpenCV
            diff = (diff * 255).astype("uint8")
            diff_box = cv2.merge([diff, diff, diff])

            # Threshold the difference image, followed by finding contours to
            # obtain the regions of the two input images that differ
            thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            mask = np.zeros(before.shape, dtype='uint8')
            filled_after = after.copy()

            for c in contours:
                area = cv2.contourArea(c)
                if area > sensitivity:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
                    cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
                    cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36, 255, 12), 2)
                    cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
                    cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

            beforergb = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
            afterrgb = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)
            diffrgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
            diff_boxrgb = cv2.cvtColor(diff_box, cv2.COLOR_BGR2RGB)
            maskrgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            filled_afterrgb = cv2.cvtColor(filled_after, cv2.COLOR_BGR2RGB)
            # Display the result image using streamlit
            result_image = Image.fromarray(afterrgb)
            st.image(result_image, caption='Result photo')

            # After displaying the result, remove the temporary files
            os.unlink(temp_file1.name)
            os.unlink(temp_file2.name)
