import streamlit as st
import logging
from keras_vggface.utils import preprocess_input
from keras_vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import cv2
from mtcnn import MTCNN
import os

logging.basicConfig(level=logging.DEBUG)

# Initialize the MTCNN face detector
detector = MTCNN()

# Load the VGGFace model
model = VGGFace(model="resnet50", include_top=False, input_shape=(224, 224, 3), pooling="avg")

# Load precomputed features and filenames
features_list = pickle.load(open("embedding.pkl", "rb"))
filenames = pickle.load(open("filesname.pkl", "rb"))

def save_upload_image(uploading_image):
    try:
        with open(os.path.join("upload", uploading_image.name), "wb") as f:
            f.write(uploading_image.getbuffer())
        return True
    except Exception as e:
        logging.error(f"Error saving uploaded image: {e}")
        return False

def extract_features(img_path, model, detector):
    try:
        img = cv2.imread(img_path)
        results = detector.detect_faces(img)

        if not results:
            st.error("No faces detected in the image.")
            return None

        x, y, width, height = results[0]['box']
        face_img = img[y:y + height, x:x + width]

        image = Image.fromarray(face_img)
        image = image.resize((224, 224))

        face_array = np.asarray(image)
        face_array = face_array.astype('float32')

        expended_array = np.expand_dims(face_array, axis=0)
        preprocessed = preprocess_input(expended_array)
        result = model.predict(preprocessed).flatten()
        return result
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return None

def predict_img(features_list, features):
    similarity = []
    for i in range(len(features_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), features_list[i].reshape(1, -1))[0][0])
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

st.title("Upload Image And Predict Similar Bollywood Actor Image")

uploading_image = st.file_uploader("Upload Your image here")
if uploading_image is not None:
    if save_upload_image(uploading_image):
        img_read = Image.open(uploading_image)
        new_image = img_read.resize((300, 300))
        st.image(new_image)

        features = extract_features(os.path.join('upload', uploading_image.name), model, detector)
        if features is not None:
            index_pos = predict_img(features_list, features)

            col1, col2 = st.beta_columns(2)

            predict_like = " ".join(filenames[index_pos].split('//')[1].split('_'))
            with col1:
                st.header("Your Uploaded Image")
                st.image(new_image)
            with col2:
                st.header("Seems like " + predict_like)
                st.image(filenames[index_pos], width=300)


