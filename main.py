import streamlit as st
import numpy as np
import cv2
from model import MyModel

st.set_page_config(layout="wide")
st.title('Is Omar in this picture?')

with st.spinner("Loading Model..."):
    model = MyModel()

## utils
def create_opencv_image_from_stringio(img_stream, cv2_img_flag=0):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

control, view = st.columns(2)
img_view = None
## control section
with control:
    
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])
    sensitivity = st.slider('Facial recognition sensitivity. Higher res requires higher sensitivity', 1, 100, 30)
    if uploaded_file is not None:
        
        img = create_opencv_image_from_stringio(uploaded_file, cv2_img_flag=1)
        
        # Converting image to grayscale  
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Loading the required haar-cascade xml classifier file
        haar_cascade = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml")
    
        # Applying the face detection method on the grayscale image
        faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1,
            minNeighbors=sensitivity,
            minSize=(100,100),
            )

        st.write("Found {0} Face(s)!".format(len(faces_rect)))
        
        img_view = img.copy()
        for (x, y, w, h) in faces_rect:
                cv2.rectangle(img_view, (x, y), (x+w, y+h), (255, 255, 255), thickness=2)
        
        st.write("Is Omar in this picture?")
        if st.button('Predict'):
            success = 0
            for (x, y, w, h) in faces_rect:
                face = img[y-sensitivity*2:y+h+sensitivity*2, x-sensitivity*2:x+w+sensitivity*2, :]
                
                # from model import preprocess
                # new = preprocess(face)
                # st.write(new.shape)
                # st.image(new, caption='Detected faces', use_column_width=True)

                # import os
                # import uuid
                # imgname =  str(uuid.uuid1()) + ".jpg"
                cv2.imwrite('input_img.jpg', face)
                
                prediction = model.predict(face)
                if prediction:
                    success += 1
                    cv2.rectangle(img_view, (x, y), (x+w, y+h), (0, 255, 0), thickness=4)
                else:
                    cv2.rectangle(img_view, (x, y), (x+w, y+h), (0, 0, 255), thickness=4)

        
with view:
    if img_view is not None:
        st.image(img_view, caption='Detected faces', use_column_width=True, channels='BGR')

        # st.image(img, caption='Uploaded Image.', use_column_width=True, channels='BGR') 
        # with st.spinner("Predicting..."):
        #     if model.predict(img):
        #         st.success("Omar is in the picture!")
        #     else:
        #         st.error("Omar is not in the picture!")