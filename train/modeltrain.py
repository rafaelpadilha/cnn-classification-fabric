import pickle
import tensorflow as tf
import numpy as np
import keras
from keras.initializers import glorot_normal
from sklearn.model_selection import train_test_split
from time import localtime, strftime
from datetime import datetime
from settings import logger, app_cfg
from train.trainingplot import TrainingPlot
from train.confusionmatrix import ConfusionMatrix
from sklearn import metrics

class ModelTrain():
    def __init__(self, model_name, batch_size, epochs, test_size=0.2, model_arq=None):
        logger.debug(f"Initializating train of model {model_name}.")
        self.RUN = strftime("%Y-%m-%d_%H:%M:%S", localtime())
        self.batch_size = batch_size
        self.epochs = epochs
        self.set_model(model_arq)
        self.model_name = model_name
        self.logdir = f"{app_cfg['path']}/logs/{self.model_name + self.RUN}"
        logger.debug("Seting up Tensorflow configurations.")
        # Tensorflow Configurations
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.85)
        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        self.tb = keras.callbacks.TensorBoard(log_dir= self.logdir)

        self.load_data(data_path =app_cfg['data_path'],test_size=test_size)
        
    def run(self):
        self.model_train(batch_size=self.batch_size, epochs=self.epochs)

    def log_confusion_matrix(self):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = self.model.predict(self.test_data)
        test_pred = np.argmax(test_pred_raw, axis=1)

        logger.debug(test_pred)
        logger.debug(self.test_label)
        # Calculate the confusion matrix.
        cm = metrics.confusion_matrix(self.test_label, test_pred)
        # Save confusion matrix.
        ConfusionMatrix(logdir=self.logdir).save_cm(cm, class_names=app_cfg['class_names'])


    def load_data(self, data_path, test_size=0.15):
        logger.info("Loading data.")
        try:
            data = pickle.load(open(data_path, "rb"))
        except Exception as err:
            #TODO Exception
            logger.error(err)
            exit
        
        logger.debug(f"Loaded {len(data)} samples.")
        X = []
        y = []
        for feature, label in data:
            X.append(feature)
            y.append(label)
        
        X = np.array(X).reshape(-1, app_cfg['img_size'], app_cfg['img_size'], 3)
        X = X/255.0

        y = keras.utils.to_categorical(y, num_classes=None)

        logger.info('Splitting dataset.')
        #TODO find another way to split dataset, using too much memory.
        self.train_data, self.test_data, self.train_label, self.test_label = train_test_split(X, y, test_size=test_size, random_state=1)
        logger.info(f'Data load complete. (Train_data:{len(self.train_data)}; Test_data:{len(self.test_data)})')

    def set_model(self, model):
        logger.info("Setting model.")
        self.model = model
        logger.debug(f"Model set, summary: {self.model.summary()}")

    def model_train(self, batch_size, epochs):
        logger.info(f"Starting model training. [batch_size={batch_size}, epochs={epochs}]")
        start_time = datetime.now()
        history = self.model.fit(self.train_data, self.train_label, batch_size=batch_size, validation_data=(self.test_data, self.test_label), epochs=epochs, callbacks=[self.tb])
        end_time = datetime.now()
        logger.info(f"Model train ended. (Train time:{end_time - start_time})")
        logger.info("Saving model.")
        try:
            self.model.save(f"{self.logdir}/model.h5")
            logger.info(f"Model saved.")
        except Exception as err:
            logger.error(err)
    
        logger.info("Plotting accuracy/loss graphics.")
        TrainingPlot(outputdir=self.logdir, history=history)
        #self.log_confusion_matrix()