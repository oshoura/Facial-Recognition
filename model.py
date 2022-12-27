import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer


# creating the distance layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    # performing similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

def preprocess(image:np.array):
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = tf.convert_to_tensor(img)
    img = tf.image.resize(img, (100,100))
    img = img/255
    return np.array(img)

class MyModel():
    
    def __init__(self, detection_threshold = 0.5, verification_threshold=0.7) -> None:
        self.model = tf.keras.models.load_model('siamese_model.h5', custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.keras.losses.BinaryCrossentropy})
        self.validation_img_path = 'verification_data'
        self.detection_threshold = detection_threshold
        self.verfication_threshold = verification_threshold
        
    def predict(self, image):
        results = []
        for num_validation, validation_image in enumerate([f for f in os.listdir(self.validation_img_path) if not f.startswith('.')]):
            # read validation image
            validation_img = cv2.imread(os.path.join(self.validation_img_path, validation_image))
            # preprocess validation image
            validation_img = preprocess(validation_img)
            
            # preprocess input image
            input_img = preprocess(image)
            
            # predict
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)), verbose = 0)[0]
            results.append(result)
            print(f"Prediction for image {num_validation}: {result}")
        
        num_of_positives = np.sum(np.array(results) > self.detection_threshold)
        verified = (num_of_positives/(num_validation+1)) > self.verfication_threshold
        # print(verified)
        
        return verified
    
# model = MyModel()
# print(model.predict(cv2.imread('input_image.jpg')))
