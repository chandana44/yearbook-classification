from keras.layers import Activation, Convolution2D
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Input
from keras.layers import merge
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.models import Model

from customlayers import crosschannelnormalization, splittensor

from util import *
from keras.optimizers import SGD


def alexnet_model(img_rows, img_cols, channels=1, num_classes=None, use_pretraining=True,
                pretrained_weights_path=None, optimizer=None, loss=None,
                fine_tuning_method=END_TO_END_FINE_TUNING):

    inputs = Input(shape=(channels, img_rows, img_cols))
    conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge([
                       Convolution2D(128, 5, 5, activation='relu', name='conv_2_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_2)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_2')

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = merge([
                       Convolution2D(192, 3, 3, activation='relu', name='conv_4_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_4)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_4')

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = merge([
                       Convolution2D(128, 3, 3, activation='relu', name='conv_5_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_5)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_5')

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)
    dense_1 = Flatten(name='flatten')(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000, name='dense_3')(dense_3)
    prediction = Activation('softmax', name='softmax')(dense_3)

    model = Model(input=inputs, output=prediction)
    if use_pretraining:
        if pretrained_weights_path:
            model.load_weights(pretrained_weights_path)
        else:
            raise Exception('use_pretraining is true but pretrained_weights_path is not specified!')

    # Removing the final dense_3 layer and adding the layers with correct classification size
    model.layers.pop()
    model.layers.pop()

    last = model.layers[-1].output
    last = Dense(NUM_CLASSES, name='dense_3')(last)
    prediction = Activation('softmax', name='softmax')(last)

    model = Model(model.input, prediction)

    if fine_tuning_method == FREEZE_INITIAL_LAYERS:
        print(get_time_string() + 'Freezing initial 5 layers of the network..')
        for layer in model.layers[:5]:
           layer.trainable = False

    if optimizer == 'sgd':
        optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    if loss == 'l1':
        loss = get_l1_loss

    print(get_time_string() + 'Compiling the model..')
    model.compile(optimizer=optimizer, loss=loss)

    print(model.summary())

    return model
