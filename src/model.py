from keras import applications
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Convolution2D
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Input
from keras.layers import merge
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.models import Model
from keras.models import load_model

from customlayers import crosschannelnormalization, splittensor
from util import *


ALEXNET_ARCHITECTURE = 'alexnet'
VGG16_ARCHITECTURE = 'vgg16'
VGG19_ARCHITECTURE = 'vgg19'
RESNET_ARCHITECTURE = 'resnet'

ARCHITECTURES = [ALEXNET_ARCHITECTURE, VGG16_ARCHITECTURE, VGG19_ARCHITECTURE, RESNET_ARCHITECTURE]

END_TO_END_FINE_TUNING = 'end-to-end'
PHASE_BY_PHASE_FINE_TUNING = 'phase-by-phase'

FINE_TUNING_METHODS = [END_TO_END_FINE_TUNING, PHASE_BY_PHASE_FINE_TUNING]


class YearbookModel:
    FINE_TUNING_METHODS = ['end-to-end', 'phase-by-phase']
    get_model_function = {}

    def __init__(self):
        self.get_model_function[ALEXNET_ARCHITECTURE] = self.getAlexNet
        self.get_model_function[VGG16_ARCHITECTURE] = self.getVGG16

    def getModel(self, model_architecture='alexnet', load_saved_model=False, model_save_path=None, use_pretraining=False,
                 pretrained_weights_path=None, train_dir=None, val_dir=None, fine_tuning_method=END_TO_END_FINE_TUNING):

        """

        :param model_architecture: which architecture to use
        :param load_saved_model: boolean (whether to just load the model from weights path)
        :param model_save_path: (final model weights path, if load_pretrained is true)
        :param pretrained_weights_path: if load_trained is false and if use_pretraining is true, the path of weights to load for pre-training
        :param train_dir: training data directory
        :param val_dir: validation data directory
        :param use_pretraining: boolean, whether to use pre-training or train from scratch
        :param fine_tuning_method: whether to use end-to-end pre-training or phase-by-phase pre-training
        :return: Returns the corresponding deepnet model

        """

        if model_architecture not in ARCHITECTURES:
            raise Exception('Invalid architecture name!')

        return self.get_model_function[model_architecture](load_saved_model, model_save_path,
                                                           use_pretraining,
                                                           pretrained_weights_path,
                                                           train_dir, val_dir,
                                                           fine_tuning_method)

    def getAlexNet(self, load_saved_model, model_save_path, use_pretraining, pretrained_weights_path,
                   train_dir, val_dir, fine_tuning_method):
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

        print(get_time_string() + 'Creating AlextNet model..')

        if load_saved_model:
            if model_save_path is None:
                raise Exception('Unable to load trained model as model_weights_path is None!')
            model = load_model(model_save_path)
            return model

        train_data = listYearbook(True, False)
        valid_data = listYearbook(False, True)

        train_images, train_labels = get_data_and_labels(train_data, YEARBOOK_TRAIN_PATH)
        valid_images, valid_labels = get_data_and_labels(valid_data, YEARBOOK_VALID_PATH)

        # Preprocessing images
        processed_train_images = preprocess_image_batch(image_paths=train_images, img_size=(256, 256),
                                                        crop_size=(227, 227), color_mode="rgb")
        processed_valid_images = preprocess_image_batch(image_paths=valid_images, img_size=(256, 256),
                                                        crop_size=(227, 227), color_mode="rgb")

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
        last = Dense(120, name='dense_3')(last)
        prediction = Activation('softmax', name='softmax')(last)

        model = Model(model.input, prediction)

        print(get_time_string() + 'Compiling the model..')
        model.compile(optimizer="sgd", loss='mse')

        print(get_time_string() + 'Fitting the model..')
        model.fit(x=processed_train_images, y=train_labels, batch_size=128, epochs=10, verbose=1,
                  validation_data=(processed_valid_images, valid_labels))

        print(get_time_string() + 'Fitting complete. Returning model..')

        if model_save_path is not None:
            print(get_time_string() + 'Saving model weights to ' + model_save_path + '..')
            model.save(model_save_path)

        return model

    def get_l1_loss(self, x, y):
        return abs(np.argmax(x) - np.argmax(y))

    def getVGG16(self, load_saved_model, model_save_path, use_pretraining, pretrained_weights_path, train_dir,
                 val_dir, fine_tuning_method):
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

        batch_size = 16
        img_height = 224
        img_width = 224
        num_epochs = 10
        samples_per_epoch = 2000
        nb_val_samples = 800

        model = applications.VGG16(weights=None, classes=120)

        model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      loss="mean_absolute_error")

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size)

        validation_generator = test_datagen.flow_from_directory(
            val_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size)

        # fine-tune the model
        model.fit_generator(
            train_generator,
            samples_per_epoch=samples_per_epoch,
            epochs=num_epochs,
            validation_data=validation_generator,
            nb_val_samples=nb_val_samples)

        return None
