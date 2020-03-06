from settings import logger, app_cfg
#
import pickle
import tensorflow as tf
import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from time import localtime, strftime

RUN = strftime("%Y-%m-%d_%H:%M:%S", localtime())

class ModelTrain():
    def __init__(self, model_name):
        logger.debug(f"Starting train of model {model_name}.")
        self.model_name = model_name
        logger.debug("Seting up Tensorflow configurations.")
        # Tensorflow Configurations
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.85)
        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        self.tb = TensorBoard(log_dir=f"/home/rafael/Desktop/python/data/simpsons/logs/{self.model_name + RUN}")

        self.datapath = app_cfg['data_path']

        self.load_data()

    def run(self):
        raise NotImplemented

    def load_data(self):
        logger.info("Loading data.")
        try:
            data = pickle.load(open(self.datapath, "rb"))
        except Exception as err:
            #TODO Exception
            logger.erro(err.msg())
            exit
        
        logger.debug(f"Loaded {len(data)} samples.")
        X = []
        y = []
        for feature, label in data:
            X.append(feature)
            y.append(label)
        
        X = np.array(X).reshape(-1, app_cfg['img_size'], app_cfg['img_size'], 3)
        X = X/255.0
        y = to_categorical(y, num_classes=None)

        logger.info('Splitting dataset.')
        self.train_data, self.test_data, self.train_label, self.test_label = train_test_split(X, y, test_size=0.2, random_state=1)
        logger.info('Data load complete.')




"""
NAME = "Simpsons-cnn-64x2-" + RUN

n_classes = 42

X = pickle.load(
    open("/home/rafael/Desktop/python/data/simpsons/X.pickle", "rb"))
y = pickle.load(
    open("/home/rafael/Desktop/python/data/simpsons/y.pickle", "rb"))

X = X/255.0

y = to_categorical(y, num_classes=None)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(X, y, batch_size=32, validation_split=0.10,
          epochs=100, callbacks=[tb])
model.save('/home/rafael/Desktop/python/data/simpsons/models/'+NAME+'.model')
"""