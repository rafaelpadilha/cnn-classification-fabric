import cv2
import tensorflow as tf
from numpy import argmax
from settings import app_cfg

import keras
from keras.initializers import glorot_normal

classes = app_cfg['class_names']

def crop_center(img, cropx, cropy):
    y, x, z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

def prepare(filepath):
    img_size = app_cfg['img_size']
    img_array = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img_array = crop_center(img_array, img_size, img_size)
    return img_array.reshape(-1, img_size, img_size, 3)


model = tf.keras.models.load_model(
    "/home/rafael/ifb/tcc/cnn-classification-fabric/logs/fabric-teste2020-03-08_23:55:50/model.h5")

prediction = model.predict_proba(
    [tf.cast(prepare("/home/rafael/Pictures/tafeta_teste.jpg"), tf.float32)])

print(classes[argmax(prediction[0])])
