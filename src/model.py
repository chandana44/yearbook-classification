from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from alexnet import alexnet_model
from densenet169 import densenet169_model
from resnet_152 import resnet152_model
from util import *
from vgg16 import vgg16_model


class YearbookModel:
    get_model_function = {}

    def __init__(self):
        self.get_model_function[ALEXNET_ARCHITECTURE] = self.getAlexNet
        self.get_model_function[VGG16_ARCHITECTURE] = self.getVGG16
        self.get_model_function[RESNET152_ARCHITECTURE] = self.getResNet152
        self.get_model_function[DENSENET169_ARCHITECTURE] = self.getDenseNet169

    def getCheckpointer(self, model_save_path):
        ext = '.h5'
        path_wo_ext = model_save_path.split(ext)[0]
        filepath = path_wo_ext + '-{epoch:02d}-{val_loss:.2f}' + ext

        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                       save_best_only=False,
                                       save_weights_only=False)
        return checkpointer

    def getModel(self, model_architecture='alexnet', load_saved_model=False, model_save_path=None,
                 use_pretraining=False,
                 pretrained_weights_path=None, train_dir=None, val_dir=None, fine_tuning_method=END_TO_END_FINE_TUNING,
                 batch_size=128, num_epochs=10, optimizer='sgd', loss='mse', initial_epoch=0, sample=False):

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
        train_data = listYearbook(True, False, sample)
        valid_data = listYearbook(False, True, sample)

        train_images, train_labels = get_data_and_labels(train_data, YEARBOOK_TRAIN_PATH)
        valid_images, valid_labels = get_data_and_labels(valid_data, YEARBOOK_VALID_PATH)

        # Preprocessing images
        print 'Preprocessing images...'
        processed_train_images = preprocess_image_batch(image_paths=train_images, architecture=model_architecture)
        processed_valid_images = preprocess_image_batch(image_paths=valid_images, architecture=model_architecture)

        return self.get_model_function[model_architecture](processed_train_images, train_labels, processed_valid_images,
                                                           valid_labels,
                                                           load_saved_model,
                                                           model_save_path,
                                                           use_pretraining,
                                                           pretrained_weights_path,
                                                           train_dir, val_dir,
                                                           fine_tuning_method,
                                                           batch_size, num_epochs,
                                                           optimizer, loss,
                                                           initial_epoch)

    def getAlexNet(self, processed_train_images, train_labels, processed_valid_images, valid_labels, load_saved_model,
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
        :param initial_epoch: starting epoch to start training
        :return: Returns the AlexNet model according to the parameters provided

        """

        print(get_time_string() + 'Creating AlexNet model..')

        img_rows, img_cols = 227, 227  # Resolution of inputs
        channels = 3

        if load_saved_model:
            if model_save_path is None:
                raise Exception('Unable to load trained model as model_save_path is None!')
            print(get_time_string() + 'Loading saved model from ' + model_save_path + '..')
            model = load_model(model_save_path)
        else:
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

    def getVGG16(self, processed_train_images, train_labels, processed_valid_images, valid_labels, load_saved_model,
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
        :param initial_epoch: starting epoch to start training
        :return: Returns the AlexNet model according to the parameters provided

        """

        print(get_time_string() + 'Creating VGG16 model..')

        img_rows, img_cols = 224, 224  # Resolution of inputs
        channels = 3

        if load_saved_model:
            if model_save_path is None:
                raise Exception('Unable to load trained model as model_save_path is None!')
            print(get_time_string() + 'Loading saved model from ' + model_save_path + '..')
            model = load_model(model_save_path)
        else:
            model = vgg16_model(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=NUM_CLASSES,
                                use_pretraining=use_pretraining, pretrained_weights_path=pretrained_weights_path,
                                optimizer=optimizer, loss=loss, fine_tuning_method=fine_tuning_method)

        if initial_epoch >= num_epochs:
            print(get_time_string() + 'Not fitting the model since initial_epoch is >= num_epochs. Returning model..')
            return model

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

    def getResNet152(self, processed_train_images, train_labels, processed_valid_images, valid_labels, load_saved_model,
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
        :param initial_epoch: starting epoch to start training
        :return: Returns the AlexNet model according to the parameters provided

        """
        print(get_time_string() + 'Creating ResNet152 model..')

        img_rows, img_cols = 224, 224  # Resolution of inputs
        channels = 3

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

    def getDenseNet169(self, processed_train_images, train_labels, processed_valid_images, valid_labels,
                       load_saved_model,
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
        :param initial_epoch: starting epoch to start training
        :return: Returns the AlexNet model according to the parameters provided

        """

        print(get_time_string() + 'Creating DenseNet169 model..')

        img_rows, img_cols = 224, 224  # Resolution of inputs
        channels = 3

        if load_saved_model:
            if model_save_path is None:
                raise Exception('Unable to load trained model as model_save_path is None!')
            print(get_time_string() + 'Loading saved model from ' + model_save_path + '..')
            model = load_model(model_save_path)
        else:
            model = densenet169_model(img_rows=img_rows, img_cols=img_cols, channels=channels,
                                      num_classes=NUM_CLASSES, use_pretraining=use_pretraining,
                                      pretrained_weights_path=pretrained_weights_path,
                                      optimizer=optimizer, loss=loss,
                                      fine_tuning_method=fine_tuning_method)

        if initial_epoch >= num_epochs:
            print(get_time_string() + 'Not fitting the model since initial_epoch is >= num_epochs. Returning model..')
            return model
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
