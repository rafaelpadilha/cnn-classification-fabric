import matplotlib.pyplot as plt
from matplotlib import use
use('agg')



class TrainingPlot():

    def __init__(self, outputdir, history):

        plt.subplot(2, 1, 1)
        plt.title('Model Evaluation')
        #accuracy
        plt.ylabel('Accuracy')
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.legend(['Train', 'Test'], loc='upper left')

        #loss
        plt.subplot(2, 1, 2)
        plt.xlabel('Epoch(s)')
        plt.ylabel('Loss')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(outputdir + "/eval.png")
        plt.clf()