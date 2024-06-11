import os
import pickle
# actors=os.listdir('data')
#
# filesname=[]
#
# for actor in actors:
#     for file in os.listdir(os.path.join('data',actor)):
#         filesname.append(os.path.join('data',actor,file))
#     # print(filesname)
#     # print(len(filesname))
# pickle.dump(filesname,open('filesname.pkl','wb'))

from keras_vggface.utils import preprocess_input
from tensorflow.keras.preprocessing import image
from keras_vggface.vggface import VGGFace
import numpy as np
from tqdm import tqdm
filesname = pickle.load(open('filesname.pkl','rb'))
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

def features_extract(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expended_img=np.expand_dims(img_array,axis=0)
    preprocessed=preprocess_input(expended_img)

    result=model.predict(preprocessed).flatten()
    return result

features=[]
for file in tqdm(filesname):
    features.append(features_extract(file,model))


pickle.dump(features,open('embedding.pkl','wb'))

