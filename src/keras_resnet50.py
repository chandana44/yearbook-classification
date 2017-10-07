from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.optimizers import SGD, Adam

from util import *


def keras_resnet50_model(img_rows=224, img_cols=224, channels=3, num_classes=400, optimizer='sgd',
                         loss='categorical_crossentropy', fine_tuning_method=END_TO_END_FINE_TUNING,
                         learning_rate=None):

    base_model = ResNet50(weights='imagenet', include_top=False,
                          input_shape=(channels, img_rows, img_cols))

    x_fc = Flatten()(base_model.output)
    predictions = Dense(num_classes, activation='softmax')(x_fc)

    model = Model(input=base_model.input, output=predictions)

    if fine_tuning_method == FREEZE_INITIAL_LAYERS:
        print(get_time_string() + 'Freezing initial 100 layers of the network..')
        for layer in model.layers[:100]:
            layer.trainable = False

    # Specifying learning rate for optimizers
    if optimizer == 'sgd':
        lr = 1e-3
        if learning_rate is not None:
            lr = learning_rate
        optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    if optimizer == 'adam':
        lr = 1e-4
        if learning_rate is not None:
            lr = learning_rate
        optimizer = Adam(lr=lr)

    if loss == 'l1':
        loss = get_l1_loss

    model.compile(optimizer=optimizer, loss=loss)

    print('Number of layers: ', len(model.layers))
    print(model.summary())

    return model
