from settings import logger, app_cfg
from time import time
from train.modeltrain import ModelTrain
from tensorflow.keras.models import model_from_json
from models.modelgenerator import save_model_json

"""
import cv2
import tensorflow as tf
from numpy import argmax

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
"""

if __name__ == '__main__':
    learning_rate = 1
    decay_learning_rate = 0.00
    validation_p = 0.15
    batch_size = 64
    epochs = 15
    model_name = "bilinear"

    save_model_json(size_height=150, size_width=150,
                    no_class=app_cfg['class_n'], lr=learning_rate, model_name=model_name)

    arq = model_from_json(
        open(f"{app_cfg['path']}models/{model_name}/model.json", 'r').read())
    #arq.load_weights(f"{app_cfg['path']}models/{model_name}/weights.h5")

    model = ModelTrain(model_name=f"adam-fabric-bcnn_val{int(validation_p*100)}-ep{epochs}-bs{batch_size}-lr{int(learning_rate*100)}_",
                       test_size=validation_p, model_arq=arq, batch_size=batch_size, epochs=epochs, lr=learning_rate, dlr=decay_learning_rate)

    model.compile()

    model.run()

    """
    prediction = model.model.predict(
    [prepare("/home/rafael/Pictures/tela2.jpg")])
    classes = app_cfg['class_names']
    print(classes[argmax(prediction[0])])
    """
