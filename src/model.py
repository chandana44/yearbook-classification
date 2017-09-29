from keras import backend as K
from keras.layers import Activation, Convolution2D
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Input
from keras.layers import merge
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.models import Model
from keras.models import load_model

from customlayers import crosschannelnormalization, splittensor
from resnet_152 import resnet152_model
from vgg16 import vgg16_model
from densenet169 import densenet169_model
from util import *

END_TO_END_FINE_TUNING = 'end-to-end'
PHASE_BY_PHASE_FINE_TUNING = 'phase-by-phase'

FINE_TUNING_METHODS = [END_TO_END_FINE_TUNING, PHASE_BY_PHASE_FINE_TUNING]


class YearbookModel:
    FINE_TUNING_METHODS = ['end-to-end', 'phase-by-phase']
    get_model_function = {}

    def __init__(self):
        self.get_model_function[ALEXNET_ARCHITECTURE] = self.getAlexNet
        self.get_model_function[VGG16_ARCHITECTURE] = self.getVGG16
        self.get_model_function[RESNET152_ARCHITECTURE] = self.getResNet152
        self.get_model_function[DENSENET169_ARCHITECTURE] = self.getDenseNet169

    def get_l1_loss(self, x, y):
        return abs(K.argmax(x) - K.argmax(y))

    def getModel(self, model_architecture='alexnet', load_saved_model=False, model_save_path=None,
                 use_pretraining=False,
                 pretrained_weights_path=None, train_dir=None, val_dir=None, fine_tuning_method=END_TO_END_FINE_TUNING,
                 batch_size=128, num_epochs=10, optimizer='sgd', loss='mse'):

        """

        :param model_architecture: which architecture to use
        :param load_saved_model: boolean (whether to just load the model from weights path)
        :param model_save_path: (final model weights path, if load_pretrained is true)
        :param pretrained_weights_path: if load_trained is false and if use_pretraining is true, the path of weights to load for pre-training
        :param train_dir: training data directory
        :param val_dir: validation data directory
        :param use_pretraining: boolean, whether to use pre-training or train from scratch
        :param fine_tuning_method: whether to use end-to-end pre-training or phase-by-phase pre-training
        :param batch_size: batch_size to use while fitting the model
        :param num_epochs: number of epochs to train the model
        :param optimizer: type of optimizer to use (sgd|adagrad)
        :param loss: type of loss to use (mse|l1)
        :return: Returns the corresponding deepnet model

        """

        if model_architecture not in ARCHITECTURES:
            raise Exception('Invalid architecture name!')

        if load_saved_model:
            if model_save_path is None:
                raise Exception('Unable to load trained model as model_weights_path is None!')
            model = load_model(model_save_path)
            return model

        # get train and validation data
        train_data = listYearbook(True, False)
        valid_data = listYearbook(False, True)

        train_images, train_labels = get_data_and_labels(train_data, YEARBOOK_TRAIN_PATH)
        valid_images, valid_labels = get_data_and_labels(valid_data, YEARBOOK_VALID_PATH)

        return self.get_model_function[model_architecture](train_images, train_labels, valid_images, valid_labels,
                                                           model_save_path,
                                                           use_pretraining,
                                                           pretrained_weights_path,
                                                           train_dir, val_dir,
                                                           fine_tuning_method,
                                                           batch_size, num_epochs,
                                                           optimizer, loss)

    def getAlexNet(self, train_images, train_labels, valid_images, valid_labels, model_save_path,
                   use_pretraining, pretrained_weights_path, train_dir, val_dir, fine_tuning_method,
                   batch_size, num_epochs, optimizer, loss):
        """

        :param load_saved_model: boolean (whether to just load the model from weights path)
        :param model_save_path: (final model weights path, if load_pretrained is true)
        :param pretrained_weights_path: if load_trained is false and if use_pretraining is true, the path of weights to load for pre-training
        :param train_dir: training data directory
        :param val_dir: validation data directory
        :param use_pretraining: boolean, whether to use pre-training or train from scratch
        :param fine_tuning_method: whether to use end-to-end pre-training or phase-by-phase pre-training
        :param batch_size: batch_size to use while fitting the model
        :param num_epochs: number of epochs to train the model
        :param optimizer: type of optimizer to use (sgd|adagrad)
        :param loss: type of loss to use (mse|l1)
        :return: Returns the AlexNet model according to the parameters provided

        """

        print(get_time_string() + 'Creating AlexNet model..')

        # Preprocessing images
        processed_train_images = preprocess_image_batch(image_paths=train_images, architecture=ALEXNET_ARCHITECTURE)
        processed_valid_images = preprocess_image_batch(image_paths=valid_images, architecture=ALEXNET_ARCHITECTURE)

        inputs = Input(shape=(3, 227, 227))
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

        print(get_time_string() + 'Compiling the model..')

        if loss == 'l1':
            model.compile(optimizer=optimizer, loss=self.get_l1_loss)
        else:
            model.compile(optimizer=optimizer, loss=loss)

        print(get_time_string() + 'Fitting the model..')
        model.fit(x=processed_train_images, y=train_labels, batch_size=batch_size, epochs=num_epochs,
                  verbose=1, validation_data=(processed_valid_images, valid_labels))

        print(get_time_string() + 'Fitting complete. Returning model..')

        if model_save_path is not None:
            print(get_time_string() + 'Saving model weights to ' + model_save_path + '..')
            model.save(model_save_path)

        return model

    def getVGG16(self, train_images, train_labels, valid_images, valid_labels, model_save_path, use_pretraining,
                 pretrained_weights_path, train_dir,
                 val_dir, fine_tuning_method, batch_size, num_epochs, optimizer, loss):
        """

        :param load_saved_model: boolean (whether to just load the model from weights path)
        :param model_save_path: (final model weights path, if load_pretrained is true)
        :param pretrained_weights_path: if load_trained is false and if use_pretraining is true, the path of weights to load for pre-training
        :param train_dir: training data directory
        :param val_dir: validation data directory
        :param use_pretraining: boolean, whether to use pre-training or train from scratch
        :param fine_tuning_method: whether to use end-to-end pre-training or phase-by-phase pre-training
        :param batch_size: batch_size to use while fitting the model
        :param num_epochs: number of epochs to train the model
        :param optimizer: type of optimizer to use (sgd|adagrad)
        :param loss: type of loss to use (mse|l1)
        :return: Returns the AlexNet model according to the parameters provided

        """

        print(get_time_string() + 'Creating VGG16 model..')

        img_rows, img_cols = 224, 224  # Resolution of inputs
        channels = 3

        # Preprocessing images
        processed_train_images = preprocess_image_batch(image_paths=train_images, architecture=VGG16_ARCHITECTURE)
        processed_valid_images = preprocess_image_batch(image_paths=valid_images, architecture=VGG16_ARCHITECTURE)

        model = vgg16_model(img_rows=img_rows, img_cols=img_cols, channel=channels, num_classes=NUM_CLASSES,
                            use_pretraining=use_pretraining, pretrained_weights_path=pretrained_weights_path,
                            optimizer=optimizer, loss=loss)

        # Start Fine-tuning
        print(get_time_string() + 'Fitting the model..')
        model.fit(processed_train_images, train_labels,
                  batch_size=batch_size,
                  nb_epoch=num_epochs,
                  shuffle=True,
                  verbose=1, validation_data=(processed_valid_images, valid_labels),
                  )

        print(get_time_string() + 'Fitting complete. Returning model..')

        if model_save_path is not None:
            print(get_time_string() + 'Saving model weights to ' + model_save_path + '..')
            model.save(model_save_path)

        return model

    def getResNet152(self, train_images, train_labels, valid_images, valid_labels, model_save_path, use_pretraining,
                     pretrained_weights_path, train_dir,
                     val_dir, fine_tuning_method, batch_size, num_epochs, optimizer, loss):
        """

        :param load_saved_model: boolean (whether to just load the model from weights path)
        :param model_save_path: (final model weights path, if load_pretrained is true)
        :param pretrained_weights_path: if load_trained is false and if use_pretraining is true, the path of weights to load for pre-training
        :param train_dir: training data directory
        :param val_dir: validation data directory
        :param use_pretraining: boolean, whether to use pre-training or train from scratch
        :param fine_tuning_method: whether to use end-to-end pre-training or phase-by-phase pre-training
        :return: Returns the AlexNet model according to the parameters provided

        """
        print(get_time_string() + 'Creating ResNet152 model..')

        img_rows, img_cols = 224, 224  # Resolution of inputs
        channels = 3

        # Preprocessing images
        processed_train_images = preprocess_image_batch(image_paths=train_images, architecture=RESNET152_ARCHITECTURE)
        processed_valid_images = preprocess_image_batch(image_paths=valid_images, architecture=RESNET152_ARCHITECTURE)

        model = resnet152_model(img_rows, img_cols, channels, NUM_CLASSES, use_pretraining, pretrained_weights_path,
                                optimizer, loss)

        # Start Fine-tuning
        print(get_time_string() + 'Fitting the model..')
        model.fit(processed_train_images, train_labels,
                  batch_size=batch_size,
                  nb_epoch=num_epochs,
                  shuffle=True,
                  verbose=1, validation_data=(processed_valid_images, valid_labels),
                  )

        print(get_time_string() + 'Fitting complete. Returning model..')

        if model_save_path is not None:
            print(get_time_string() + 'Saving model weights to ' + model_save_path + '..')
            model.save(model_save_path)

        return model

    def getDenseNet169(self, train_images, train_labels, valid_images, valid_labels, model_save_path, use_pretraining,
                 pretrained_weights_path, train_dir,
                 val_dir, fine_tuning_method, batch_size, num_epochs, optimizer, loss):
        """

        :param load_saved_model: boolean (whether to just load the model from weights path)
        :param model_save_path: (final model weights path, if load_pretrained is true)
        :param pretrained_weights_path: if load_trained is false and if use_pretraining is true, the path of weights to load for pre-training
        :param train_dir: training data directory
        :param val_dir: validation data directory
        :param use_pretraining: boolean, whether to use pre-training or train from scratch
        :param fine_tuning_method: whether to use end-to-end pre-training or phase-by-phase pre-training
        :param batch_size: batch_size to use while fitting the model
        :param num_epochs: number of epochs to train the model
        :param optimizer: type of optimizer to use (sgd|adagrad)
        :param loss: type of loss to use (mse|l1)
        :return: Returns the AlexNet model according to the parameters provided

        """

        print(get_time_string() + 'Creating DenseNet169 model..')

        img_rows, img_cols = 224, 224  # Resolution of inputs
        channels = 3

        # Preprocessing images
        processed_train_images = preprocess_image_batch(image_paths=train_images, architecture=VGG16_ARCHITECTURE)
        processed_valid_images = preprocess_image_batch(image_paths=valid_images, architecture=VGG16_ARCHITECTURE)

        model = densenet169_model(img_rows=img_rows, img_cols=img_cols, channels=channels,
                                  num_classes=NUM_CLASSES, use_pretraining=use_pretraining,
                                  pretrained_weights_path=pretrained_weights_path,
                                  optimizer=optimizer, loss=loss)

        # Start Fine-tuning
        print(get_time_string() + 'Fitting the model..')
        model.fit(processed_train_images, train_labels,
                  batch_size=batch_size,
                  nb_epoch=num_epochs,
                  shuffle=True,
                  verbose=1, validation_data=(processed_valid_images, valid_labels),
                  )

        print(get_time_string() + 'Fitting complete. Returning model..')

        if model_save_path is not None:
            print(get_time_string() + 'Saving model weights to ' + model_save_path + '..')
            model.save(model_save_path)

        return model

