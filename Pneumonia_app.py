import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

image_size = (150,150)
class_labels = ['NORMAL', 'PNEUMONIA']

@st.cache_resource
def load_Pneamonia_model():
    model = load_model('Pneumonia_detection_3.h5')
    return model

model = load_Pneamonia_model()

def preprocess_image(img: Image.Image) -> np.ndarray :
    img = img.convert('RGB')
    img = img.resize(image_size)
    image_array = img_to_array(img)
    image_array = image_array/255.0
    image_array = np.expand_dims(image_array,axis=0)
    return image_array

def predict_image(img: Image.Image):
    processed = preprocess_image(img)
    proba = model.predict(processed)[0][0]
    if proba < 0.5 :
        label = class_labels[0]
        confidence = 1-proba
    else :
        label = class_labels[1]
        confidence = proba
    
    return label,float(confidence),float(proba)

st.set_page_config(
    page_title='Pneumonia Classifier',
    page_icon= '72.png',
    layout='centered'
    
)
st.title(' X-ray Pneumonia Classifier ') 
st.write(
    'Upload a chest X-ray image and model will predict whether it is '
    '**NORMAL** or **PNEUMONIA** '  
)
st.sidebar.header('About')
st.sidebar.write(
    """ This app uses a convolutional neural network trained on the 
    **Chest X-ray Pneumonia** dataset"""
)
st.sidebar.write(" Threshold : 0.5 (sigmoid output)")
upload_file = st.file_uploader(
    'Upload a chest X-ray image (JPG/JPEG/PNG)',
    type= ['jpg','jpeg','png']
)

if upload_file is not None:
    image = Image.open(upload_file) 
    st.image(image,caption='Uploaded X-ray', width=600)
    
    if st.button('Predict'):
        with st.spinner('Running model...'):
            label,confidence,proba = predict_image(image)
        
        st.subheader('Prediction')
        if label =='PNEUMONIA':
            st.error(f'Prediction: **{label}**')
        else:
            st.success(f'Prediction: **{label}**')
        
        st.write(f'Raw model output (sigmoid): `{proba:.4f}')
        st.write(f'Confidence for predicted class : `{confidence*100:.2f}')
        
        st.progress(min(max(confidence,0.0),1.0))

else:
    st.info('Please upload a chest X-ray image to begin') 
    
        