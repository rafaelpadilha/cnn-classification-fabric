import logging


def app_cfg():
    config = {}
    config['path'] = "/home/rafael/ifb/tcc/cnn-classification-fabric/"
    config['data_path'] = config['path'] + "data/fabric.pickle"
    config['img_size'] = 150
    config['class_names'] = ['tela', 'sarja', 'cetim']
    config['class_n'] = len(config['class_names'])
    return config


logging.basicConfig(
    format=("%(asctime)s,%(msecs)-3d - %(name)-12s - %(levelname)-8s => %(message)s"),
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)
app_cfg = app_cfg()
