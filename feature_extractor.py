# from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.utils.image_utils import img_to_array
import numpy as np

class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights="imagenet")
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)

    def extract(self, img):
        img = img.resize((224,224)).convert("RGB")
        x = img_to_array(img) # to np.array
        x = np.expand_dims(x, axis=0) # (H,W,C) -> (1,H,W,C)
        x = preprocess_input(x) # Subtract avg pixel value
        feature = self.model.predict(x)[0] # (1,4096) -> (4096,)
        return feature/np.linalg.norm(feature) # Normalize
