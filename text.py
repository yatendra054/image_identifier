from keras_vggface.utils import preprocess_input
from keras_vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import cv2
from mtcnn import MTCNN

features_list = np.array(pickle.load(open('embedding.pkl','rb')))
filesname = pickle.load(open('filesname.pkl','rb'))

model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

dectector = MTCNN()
sample_img = cv2.imread('simple/Photo.png')

results = dectector.detect_faces(sample_img)
x,y,width,height = results[0]['box']

face_img = sample_img[y:y+height,x:x+width]

image = Image.fromarray(face_img)
image = image.resize((224,224))

face_array = np.asarray(image)
face_array = face_array.astype('float32')

expended_array = np.expand_dims(face_array,axis=0)
preprocessed = preprocess_input(expended_array)
result = model.predict(preprocessed).flatten()

similarity = []
for i in range(len(features_list)):
    similarity.append(cosine_similarity(result.reshape(1, -1), features_list[i].reshape(1, -1))[0][0])
index_pos=sorted(list(enumerate(similarity)), reverse=True, key=lambda x:x[1])[0][0]

temp_img = cv2.imread(filesname[index_pos])
temp_img =cv2.resize(temp_img, (224,224))
simple_img = cv2.resize(sample_img, (224,224))
cv2.imshow('Input image', sample_img)
cv2.imshow('Predicted image', temp_img)
cv2.waitKey(0)