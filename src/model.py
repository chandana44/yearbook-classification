from keras import backend as K
from keras.models import load_model

from alexnet import alexnet_model
from densenet169 import densenet169_model
from resnet_152 import resnet152_model
from util import *
from vgg16 import vgg16_model
from keras.callbacks import ModelCheckpoint

END_TO_END_FINE_TUNING = 'end-to-end'
PHASE_BY_PHASE_FINE_TUNING = 'phase-by-phase'
FREEZE_INITIAL_LAYERS = 'freeze-initial'

FINE_TUNING_METHODS = [END_TO_END_FINE_TUNING, PHASE_BY_PHASE_FINE_TUNING, FREEZE_INITIAL_LAYERS]


class YearbookModel:
    get_model_function = {}

    def __init__(self):
        self.get_model_function[ALEXNET_ARCHITECTURE] = self.getAlexNet
        self.get_model_function[VGG16_ARCHITECTURE] = self.getVGG16
        self.get_model_function[RESNET152_ARCHITECTURE] = self.getResNet152
        self.get_model_function[DENSENET169_ARCHITECTURE] = self.getDenseNet169

    def get_l1_loss(self, x, y):
        return abs(K.argmax(x) - K.argmax(y))

    def getCheckpointer(self, model_save_path):
        ext = '.h5'
        path_wo_ext = model_save_path.split(ext)[0]
        filepath = path_wo_ext + '{epoch:02d}-{val_loss:.2f}' + ext

        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                       save_best_only=False,
                                       save_weights_only=False)
        return checkpointer

    def getModel(self, model_architecture='alexnet', load_saved_model=False, model_save_path=None,
                 use_pretraining=False,
                 pretrained_weights_path=None, train_dir=None, val_dir=None, fine_tuning_method=END_TO_END_FINE_TUNING,
                 batch_size=128, num_epochs=10, optimizer='sgd', loss='mse', initial_epoch=0):

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
        :param initial_epoch: starting epoch to start training
        :return: Returns the corresponding deepnet model

        """

        if model_architecture not in ARCHITECTURES:
            raise Exception('Invalid architecture name!')

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
                                                           optimizer, loss,
                                                           initial_epoch)

    def getAlexNet(self, train_images, train_labels, valid_images, valid_labels, load_saved_model,
                   model_save_path, use_pretraining, pretrained_weights_path, train_dir, val_dir,
                   fine_tuning_method, batch_size, num_epochs, optimizer, loss, initial_epoch):
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

        img_rows, img_cols = 224, 224  # Resolution of inputs
        channels = 3

        # Preprocessing images
        processed_train_images = preprocess_image_batch(image_paths=train_images, architecture=ALEXNET_ARCHITECTURE)
        processed_valid_images = preprocess_image_batch(image_paths=valid_images, architecture=ALEXNET_ARCHITECTURE)

        model = alexnet_model(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=NUM_CLASSES,
                              use_pretraining=use_pretraining, pretrained_weights_path=pretrained_weights_path,
                              optimizer=optimizer, loss=loss, fine_tuning_method=fine_tuning_method)

        # Start Fine-tuning
        print(get_time_string() + 'Fitting the model..')
        model.fit(processed_train_images, train_labels,
                  batch_size=batch_size,
                  nb_epoch=num_epochs,
                  shuffle=True,
                  verbose=1, validation_data=(processed_valid_images, valid_labels),
                  callbacks=[self.getCheckpointer(model_save_path)]
                  )

        print(get_time_string() + 'Fitting complete. Returning model..')

        if model_save_path is not None:
            print(get_time_string() + 'Saving model weights to ' + model_save_path + '..')
            model.save(model_save_path)

        return model

    def getVGG16(self, train_images, train_labels, valid_images, valid_labels, load_saved_model,
                 model_save_path, use_pretraining, pretrained_weights_path, train_dir,
                 val_dir, fine_tuning_method, batch_size, num_epochs, optimizer, loss, initial_epoch):
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

        if load_saved_model:
            if model_save_path is None:
                raise Exception('Unable to load trained model as model_weights_path is None!')
            model = load_model(model_save_path)
        else:
            model = vgg16_model(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=NUM_CLASSES,
                                use_pretraining=use_pretraining, pretrained_weights_path=pretrained_weights_path,
                                optimizer=optimizer, loss=loss, fine_tuning_method=fine_tuning_method)

        # Start Fine-tuning
        print(get_time_string() + 'Fitting the model..')
        model.fit(processed_train_images, train_labels,
                  batch_size=batch_size,
                  nb_epoch=num_epochs,
                  shuffle=True,
                  verbose=1, validation_data=(processed_valid_images, valid_labels),
                  callbacks=[self.getCheckpointer(model_save_path)],
                  initial_epoch=initial_epoch
                  )

        print(get_time_string() + 'Fitting complete. Returning model..')

        if model_save_path is not None:
            print(get_time_string() + 'Saving model weights to ' + model_save_path + '..')
            model.save(model_save_path)

        return model

    def getResNet152(self, train_images, train_labels, valid_images, valid_labels, load_saved_model,
                     model_save_path, use_pretraining, pretrained_weights_path, train_dir,
                     val_dir, fine_tuning_method, batch_size, num_epochs, optimizer, loss, initial_epoch):
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
                  callbacks=[self.getCheckpointer(model_save_path)]
                  )

        print(get_time_string() + 'Fitting complete. Returning model..')

        if model_save_path is not None:
            print(get_time_string() + 'Saving model weights to ' + model_save_path + '..')
            model.save(model_save_path)

        return model

    def getDenseNet169(self, train_images, train_labels, valid_images, valid_labels, load_saved_model,
                       model_save_path, use_pretraining, pretrained_weights_path, train_dir,
                       val_dir, fine_tuning_method, batch_size, num_epochs, optimizer, loss, initial_epoch):
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
        processed_train_images = preprocess_image_batch(image_paths=train_images, architecture=DENSENET169_ARCHITECTURE)
        processed_valid_images = preprocess_image_batch(image_paths=valid_images, architecture=DENSENET169_ARCHITECTURE)

        model = densenet169_model(img_rows=img_rows, img_cols=img_cols, channels=channels,
                                  num_classes=NUM_CLASSES, use_pretraining=use_pretraining,
                                  pretrained_weights_path=pretrained_weights_path,
                                  optimizer=optimizer, loss=loss,
                                  fine_tuning_method=fine_tuning_method)

        # Start Fine-tuning
        print(get_time_string() + 'Fitting the model..')
        model.fit(processed_train_images, train_labels,
                  batch_size=batch_size,
                  nb_epoch=num_epochs,
                  shuffle=True,
                  verbose=1, validation_data=(processed_valid_images, valid_labels),
                  callbacks=[self.getCheckpointer(model_save_path)]
                  )

        print(get_time_string() + 'Fitting complete. Returning model..')

        if model_save_path is not None:
            print(get_time_string() + 'Saving model weights to ' + model_save_path + '..')
            model.save(model_save_path)

        return model
