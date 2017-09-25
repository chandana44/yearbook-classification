import numpy as np
from keras import applications
from keras import optimizers
from keras.layers import Activation
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Input
from keras.layers import merge
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imread
from scipy.misc import imresize

from customlayers import *
from util import *


class Model:
    ARCHITECTURES = ['alextnet', 'vgg16']
    FINE_TUNING_METHODS = ['end-to-end', 'phase-by-phase']
    get_model_function = {}

    def __init__(self):
        self.get_model_function['alexnet'] = self.getAlexNet
        self.get_model_function['vgg16'] = self.getVGG16

    def preprocess_image_batch(image_paths, img_size=None, crop_size=None, color_mode='rgb', out=None):
        """
        Consistent preprocessing of images batches

        :param image_paths: iterable: images to process
        :param crop_size: tuple: crop images if specified
        :param img_size: tuple: resize images if specified
        :param color_mode: Use rgb or change to bgr mode based on type of model you want to use
        :param out: append output to this iterable if specified
        """
        img_list = []

        for im_path in image_paths:
            img = imread(im_path, mode='RGB')
            if img_size:
                img = imresize(img, img_size)

            img = img.astype('float32')
            # We normalize the colors (in RGB space) with the empirical means on the training set
            img[:, :, 0] -= 123.68
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 103.939
            # We permute the colors to get them in the BGR order
            if color_mode == 'bgr':
                img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
            img = img.transpose((2, 0, 1))

            if crop_size:
                img = img[:, (img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2
                , (img_size[1] - crop_size[1]) // 2:(img_size[1] + crop_size[1]) // 2]

            img_list.append(img)

        try:
            img_batch = np.stack(img_list, axis=0)
        except:
            raise ValueError('when img_size and crop_size are None, images'
                             ' in image_paths must have the same shapes.')

        if out is not None and hasattr(out, 'append'):
            out.append(img_batch)
        else:
            return img_batch

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
        train_data = listYearbook(True, False)
        valid_data = listYearbook(False, True)

        train_images = [path.join(YEARBOOK_PATH, item[0]) for item in train_data]
        train_labels = []
        for item in train_data:
            label_vec = np.zeros(120)
            label_vec[item[1]-1900] = 1
            train_labels += label_vec

        valid_images = [path.join(YEARBOOK_PATH, item[0]) for item in valid_data]
        valid_labels = []
        for item in valid_data:
            label_vec = np.zeros(120)
            label_vec[item[1]-1900] = 1
            valid_labels += label_vec

        #preprocessing images
        train_images = self.preprocess_image_batch(train_images, img_size=(256, 256), crop_size=(227, 227),
                                                   color_mode="rgb")
        valid_images = self.preprocess_image_batch(valid_images, img_size=(256, 256), crop_size=(227, 227),
                                                   color_mode="rgb")

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
        if(use_pretraining):
            if model_weights_path:
                model.load_weights(model_weights_path)

        model.layers.pop()
        model.layers.pop()
        last = model.layers[-1].output
        last = Dense(110, name='dense_3')(last)
        prediction = Activation('softmax', name='softmax')(last)
        model = Model(model.input, prediction)
        model.compile(optimizer="sgd", loss='mse')
        model.fit(train_images, train_labels, batch_size = 384, nb_epoch = 100, verbose = 1)
        return model

    def get_l1_loss(self, x, y):
        return abs(np.argmax(x)-np.argmax(y))

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

        batch_size = 16
        img_height = 224
        img_width = 224
        num_epochs = 10
        samples_per_epoch = 2000
        nb_val_samples = 800

        model = applications.VGG16(classes=120)

        model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      loss=self.get_l1_loss)

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
