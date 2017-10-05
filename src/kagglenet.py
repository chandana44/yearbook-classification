from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers


def KaggleNetModel(img_rows, img_cols, channels=1, num_classes=None, use_pretraining=True,
                   pretrained_weights_path=None, fine_tuning_method=END_TO_END_FINE_TUNING, optimizer=None,
                   loss=None):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, channels)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes))
    model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=1e-4))

    print(model.summary())

    return model
