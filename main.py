from settings import logger, app_cfg
from time import time
from train.modeltrain import ModelTrain
from tensorflow.keras.models import load_model

import keras
from keras.initializers import glorot_normal

"""
import cv2
import tensorflow as tf
from numpy import argmax
"""


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


def _outer_product(x):
    '''Calculate outer-products of two tensors.
    Args:
        x: a list of two tensors.
        Assume that each tensor has shape = (size_minibatch, total_pixels, size_filter)
    Returns:
        Outer-products of two tensors.
    '''
    return keras.backend.batch_dot(x[0], x[1], axes=[1, 1]) / x[0].get_shape().as_list()[1]


def _signed_sqrt(x):
    '''Calculate element-wise signed square-root.
    Args:
        x: input tensor.
    Returns:
        Element-wise signed square-root tensor.
    '''
    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)


def _l2_normalize(x, axis=-1):
    '''Calculate L2 normalization.
    Args:
        x: input tensor.
        axis: axis for narmalization.
    Returns:
        L2 normalized tensor.
    '''
    return keras.backend.l2_normalize(x, axis=axis)


def buil_bcnn(
        all_trainable=False,

        size_height=448,
        size_width=448,
        no_class=200,
        no_last_layer_backbone=17,

        name_optimizer='adam',
        learning_rate=1.0,
        decay_learning_rate=0.0,
        decay_weight_rate=0.0,

    name_initializer='glorot_normal',
    name_activation='softmax',
    name_loss='categorical_crossentropy'
):
    '''Build Bilinear CNN.
    Detector and extractor are both VGG16.
    Args:
        all_trainable: fix or unfix VGG16 layers.
        size_height: default 448.
        size_width: default 448.
        no_class: number of prediction classes.
        no_last_layer_backbone: number of VGG16 backbone layer.
        name_optimizer: optimizer method.
        learning_rate: learning rate.
        decay_learning_rate: learning rate decay.
        decay_weight_rate: l2 normalization decay rate.
        name_initializer: initializer method.
        name_activation: activation method.
        name_loss: loss function.
    Returns:
        Bilinear CNN model.
    '''
    ##########################
    # Load pre-trained model #
    ##########################

    # Load model
    input_tensor = keras.layers.Input(shape=[size_height, size_width, 3])
    pre_train_model = keras.applications.vgg16.VGG16(
        input_tensor=input_tensor,
        include_top=False,
        weights='imagenet')

    # Pre-trained weights
    for layer in pre_train_model.layers:
        layer.trainable = all_trainable

    ######################
    # Combine two models #
    ######################

    # Extract features form detecotr
    model_detector = pre_train_model
    output_detector = model_detector.layers[no_last_layer_backbone].output
    shape_detector = model_detector.layers[no_last_layer_backbone].output_shape

    # Extract features from extractor
    model_extractor = pre_train_model
    output_extractor = model_extractor.layers[no_last_layer_backbone].output
    shape_extractor = model_extractor.layers[no_last_layer_backbone].output_shape

    # Reshape tensor to (minibatch_size, total_pixels, filter_size)
    output_detector = keras.layers.Reshape(
        [shape_detector[1]*shape_detector[2], shape_detector[-1]])(output_detector)
    output_extractor = keras.layers.Reshape(
        [shape_extractor[1]*shape_extractor[2], shape_extractor[-1]])(output_extractor)

    # Outer-products
    x = keras.layers.Lambda(_outer_product)(
        [output_detector, output_extractor])
    # Reshape tensor to (minibatch_size, filter_size_detector*filter_size_extractor)
    x = keras.layers.Reshape([shape_detector[-1]*shape_extractor[-1]])(x)
    # Signed square-root
    x = keras.layers.Lambda(_signed_sqrt)(x)
    # L2 normalization
    x = keras.layers.Lambda(_l2_normalize)(x)

    ###############################
    # Attach full-connected layer #
    ###############################

    if name_initializer is not None:
        name_initializer = eval(name_initializer+'()')

    # FC layer
    x = keras.layers.Dense(
        units=no_class,
        kernel_initializer=name_initializer,
        kernel_regularizer=keras.regularizers.l2(decay_weight_rate))(x)
    output_tensor = keras.layers.Activation(name_activation)(x)

    #################
    # Compile model #
    #################

    model_bcnn = keras.models.Model(
        inputs=[input_tensor], outputs=[output_tensor])

    # Optimizer
    if name_optimizer == 'adam':
        optimizer = keras.optimizers.Adam(
            lr=learning_rate, decay=decay_learning_rate)
    elif name_optimizer == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(
            lr=learning_rate, decay=decay_learning_rate)
    elif name_optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, decay=decay_learning_rate, momentum=0.9, nesterov=None)
    else:
        raise RuntimeError('Optimizer should be one of Adam, RMSprop and SGD.')

    # Compile
    model_bcnn.compile(loss=name_loss, optimizer=optimizer,
                       metrics=['accuracy'])

    # print('-------- Mode summary --------')
    # print(model_bcnn.summary())
    # print('------------------------------')

    return model_bcnn


def save_model(
    size_height=448,
    size_width=448,
    no_class=200,
    lr=1.0
):
    '''Save Bilinear CNN to current directory.
    The model will be saved as `model.json`.
    Args:
        size_height: default 448.
        size_width: default 448.
        no_class: number of prediction classes.
    Returns:
        Bilinear CNN model.
    '''
    model = buil_bcnn(
        size_height=size_height,
        size_width=size_width,
        no_class=no_class,
        learning_rate=lr)

    # model.save("models/bilinear.h5")
    # TODO TRY JSON SAVE
    # Save model json
    #model_json = model.to_json()
    # with open('./model.json', 'w') as f:
    #    f.write(model_json)


    return model


if __name__ == '__main__':
    learning_rate = 1
    validation_p = 0.15
    batch_size = 64
    epochs = 15

    arq = save_model(size_height=150, size_width=150,
                     no_class=app_cfg['class_n'], lr=learning_rate)
    model = ModelTrain(model_name=f"adam-fabric-bcnn_val{int(validation_p*100)}-ep{epochs}-bs{batch_size}-lr{int(learning_rate*100)}_",
                       test_size=validation_p, model_arq=arq, batch_size=batch_size, epochs=epochs)
    model.run()

    """
    prediction = model.model.predict(
    [prepare("/home/rafael/Pictures/tela2.jpg")])
    classes = app_cfg['class_names']
    print(classes[argmax(prediction[0])])
    """
