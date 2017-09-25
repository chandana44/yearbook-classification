from convnetskeras.customlayers import crosschannelnormalization
from convnetskeras.customlayers import Softmax4D
from convnetskeras.customlayers import splittensor
from convnetskeras.imagenet_tool import synset_to_dfs_ids
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import merge
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD
from scipy.misc import imread
from scipy.misc import imresize

class Model:
    ARCHITECTURES = ['alextnet', 'vgg16']
    FINE_TUNING_METHODS = ['end-to-end', 'phase-by-phase']
    get_model_function = {}

    def __init__(self):
        self.get_model_function['alexnet'] = self.getAlexNet
        self.get_model_function['vgg16'] = self.getVGG16

    def getModel(self, model_architecture, load_trained, model_weights_path, pretrained_weights_path,
                 train_dir, val_dir, use_pretraining, fine_tuning_method):
        """

        :param model_architecture: which architecture to use
        :param load_trained: boolean (whether to just load the model from weights path)
        :param model_weights_path: (final model weights path, if load_pretrained is true)
        :param pretrained_weights_path: if load_trained is false and if use_pretraining is true, the path of weights to load for pre-training
        :param train_dir: training data directory
        :param val_dir: validation data directory
        :param use_pretraining: boolean, whether to use pre-training or train from scratch
        :param fine_tuning_method: whether to use end-to-end pre-training or phase-by-phase pre-training
        :return: Returns the corresponding deepnet model

        """

        if model_architecture not in self.ARCHITECTURES:
            raise 'Invalid architecture name!'
        return self.get_model_function[model_architecture](load_trained, model_weights_path,
                                                           pretrained_weights_path,
                                                           train_dir, val_dir, use_pretraining,
                                                           fine_tuning_method)

    def getAlexNet(self, load_trained, model_weights_path, pretrained_weights_path,
                   train_dir, val_dir, use_pretraining, fine_tuning_method):
        """

        :param load_trained: boolean (whether to just load the model from weights path)
        :param model_weights_path: (final model weights path, if load_pretrained is true)
        :param pretrained_weights_path: if load_trained is false and if use_pretraining is true, the path of weights to load for pre-training
        :param train_dir: training data directory
        :param val_dir: validation data directory
        :param use_pretraining: boolean, whether to use pre-training or train from scratch
        :param fine_tuning_method: whether to use end-to-end pre-training or phase-by-phase pre-training
        :return: Returns the AlexNet model according to the parameters provided

        """

    def getAlexNet(self, load_pretrained, weights_path, train_dir, val_dir, use_pre_training, fine_tuning_method):
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

        if weights_path:
            model.load_weights(weights_path)

        model.layers.pop()
        model.layers.pop()
        last = model.layers[-1].output
        last = Dense(110, name='dense_3')(last)
        prediction = Activation('softmax', name='softmax')(last)
        model = Model(model.input, prediction)
        model.compile(optimizer="sgd", loss='mse')
        model.fit(,, batch_size = 384, nb_epoch = NUM_EPOCHS, verbose = 1)
        return model

    def getVGG16(self, load_trained, model_weights_path, pretrained_weights_path, train_dir,
                 val_dir, use_pretraining, fine_tuning_method):
        """

        :param load_trained: boolean (whether to just load the model from weights path)
        :param model_weights_path: (final model weights path, if load_pretrained is true)
        :param pretrained_weights_path: if load_trained is false and if use_pretraining is true, the path of weights to load for pre-training
        :param train_dir: training data directory
        :param val_dir: validation data directory
        :param use_pretraining: boolean, whether to use pre-training or train from scratch
        :param fine_tuning_method: whether to use end-to-end pre-training or phase-by-phase pre-training
        :return: Returns the AlexNet model according to the parameters provided

        """

        return None
