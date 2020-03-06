from settings import logger
from time import time
from train.modeltrain import ModelTrain

if __name__=='__main__':
    #Load
    logger.info("Starting code.")
    start_time = time()
    ModelTrain(model_name="simpsons-teste1")

    #Run
    #TODO