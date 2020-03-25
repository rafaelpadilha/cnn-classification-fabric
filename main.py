from settings import logger, app_cfg
from time import time
from train.modeltrain import ModelTrain
from tensorflow.keras.models import model_from_json
from models.modelgenerator import save_model_json

if __name__ == '__main__':
    learning_rate = 0.6
    decay_learning_rate = 0.0
    validation_p = 0.2
    batch_size = 128
    epochs = 20
    model_name = "bilinear"

    save_model_json(size_height=150, size_width=150,
                    no_class=app_cfg['class_n'], lr=learning_rate, model_name=model_name)

    arq = model_from_json(
        open(f"{app_cfg['path']}models/{model_name}/model.json", 'r').read())

    arq.load_weights(f"{app_cfg['path']}models/{model_name}/weights.h5")

    model = ModelTrain(model_name=f"adam-fabric-bcnn_val{int(validation_p*100)}-ep{epochs}-bs{batch_size}-lr{int(learning_rate*100)}_",
                       test_size=validation_p, model_arq=arq, batch_size=batch_size, epochs=epochs, lr=learning_rate, dlr=decay_learning_rate)

    

    model.compile()

    model.run()
