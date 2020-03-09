import matplotlib.pyplot as plt
from matplotlib import use
use('agg')



class TrainingPlot():

    def __init__(self, outputdir, history):
        self.history = history
        self.outputdir = outputdir
        self.plot_accuracy()
        self.plot_loss()

    def plot_accuracy(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.outputdir + "/accuracy.png")
        plt.clf()

    def plot_loss(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.outputdir + "/loss.png")
        plt.clf()