from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from alexnet import alexnet_model
from densenet169 import densenet169_model
from densenet121 import densenet121_model
from densenet161 import densenet161_model
from resnet_152 import resnet152_model
from resnet_50 import resnet50_model
from util import *
from vgg16 import vgg16_model
from vgg19 import vgg19_model
import customlayers

class YearbookModel:
    get_model_function = {}

    def __init__(self):
        self.get_model_function[ALEXNET_ARCHITECTURE] = self.getAlexNet
        self.get_model_function[VGG16_ARCHITECTURE] = self.getVGG16
        self.get_model_function[VGG19_ARCHITECTURE] = self.getVGG19
        self.get_model_function[RESNET152_ARCHITECTURE] = self.getResNet152
        self.get_model_function[RESNET50_ARCHITECTURE] = self.getResNet50
        self.get_model_function[DENSENET169_ARCHITECTURE] = self.getDenseNet169
        self.get_model_function[DENSENET161_ARCHITECTURE] = self.getDenseNet161
        self.get_model_function[DENSENET121_ARCHITECTURE] = self.getDenseNet121

    def getCheckpointer(self, model_save_path):
        ext = '.h5'
        path_wo_ext = model_save_path.split(ext)[0]
        filepath = path_wo_ext + '-{epoch:02d}-{val_loss:.2f}' + ext

        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                       save_best_only=False,
                                       save_weights_only=False)
        return checkpointer

    def getCheckpointFileName(self, base_model_save_path, epoch):
        ext = '.h5'
        path_wo_ext = base_model_save_path.split(ext)[0]
        filepath = path_wo_ext + '-' + str(epoch) + ext
        return filepath

    def getWeightCheckpointFileName(self, base_model_save_path, epoch):
        ext = '.h5'
        path_wo_ext = base_model_save_path.split(ext)[0]
        filepath = path_wo_ext + '-' + str(epoch) + '-weights' + ext
        return filepath

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

        train_images, train_labels = get_data_and_labels(train_data, YEARBOOK_TRAIN_PATH)

        # Preprocessing images
        print(get_time_string() + 'Preprocessing images...')
        # processed_train_images = preprocess_image_batch(image_paths=train_images, architecture=model_architecture)

        return self.get_model_function[model_architecture](train_images, train_labels,
                                                           load_saved_model,
                                                           model_save_path,
                                                           use_pretraining,
                                                           pretrained_weights_path,
                                                           train_dir, val_dir,
                                                           fine_tuning_method,
                                                           batch_size, num_epochs,
                                                           optimizer, loss,
                                                           initial_epoch,
                                                           sample)

    def getAlexNet(self, train_images, train_labels, load_saved_model,
                   model_save_path, use_pretraining, pretrained_weights_path, train_dir, val_dir,
                   fine_tuning_method, batch_size, num_epochs, optimizer, loss, initial_epoch, sample):
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
            print(get_time_string() + 'Loading saved model weights from ' + model_save_path + '..')
            model = alexnet_model(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=NUM_CLASSES_YEARBOOK,
                                  use_pretraining=use_pretraining, pretrained_weights_path=pretrained_weights_path,
                                  optimizer=optimizer, loss=loss, fine_tuning_method=fine_tuning_method,
                                  weights_path=model_save_path)
        else:
            model = alexnet_model(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=NUM_CLASSES_YEARBOOK,
                                  use_pretraining=use_pretraining, pretrained_weights_path=pretrained_weights_path,
                                  optimizer=optimizer, loss=loss, fine_tuning_method=fine_tuning_method)

        if initial_epoch >= num_epochs:
            print(get_time_string() + 'Not fitting the model since initial_epoch is >= num_epochs. Returning model..')
            return model

        # Start Fine-tuning
        print(get_time_string() + 'Fitting the model..')
        for e in range(initial_epoch, num_epochs):
            print_line()
            print('Starting epoch ' + str(e))
            print_line()
            completed = 0

            for x_chunk, y_chunk in chunks(train_images, train_labels, batch_size, ALEXNET_ARCHITECTURE):
                print(get_time_string() + 'Fitting model for chunk of size ' + str(len(x_chunk)) + '...')
                model.fit(x_chunk, y_chunk,
                          batch_size=batch_size,
                          nb_epoch=1,
                          verbose=1
                          )
                completed += len(x_chunk)
                print(get_time_string() + str(completed) + ' of ' + str(len(train_images)) + ' complete. ')

            # Not saving model since there's some bug while loading from saved model
            # file_name = self.getCheckpointFileName(base_model_save_path=model_save_path, epoch=e)
            # print(get_time_string() + 'Saving model to ' + file_name)
            # model.save(file_name)

            file_name = self.getWeightCheckpointFileName(base_model_save_path=model_save_path, epoch=e)
            print(get_time_string() + 'Saving model weights to ' + file_name)
            model.save_weights(file_name)

            print(get_time_string() + 'Epoch ' + str(e) + ' complete. Evaluating on validation set..')
            evaluateYearbookFromModel(model=model, architecture=ALEXNET_ARCHITECTURE, sample=sample)

            print_line()

        print(get_time_string() + 'Fitting complete. Returning model..')

        if model_save_path is not None:
            print(get_time_string() + 'Saving final model to ' + model_save_path + '..')
            model.save(model_save_path)

        return model

    def getVGG16(self, train_images, train_labels, load_saved_model,
                 model_save_path, use_pretraining, pretrained_weights_path, train_dir,
                 val_dir, fine_tuning_method, batch_size, num_epochs, optimizer, loss, initial_epoch, sample):
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
            model = vgg16_model(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=NUM_CLASSES_YEARBOOK,
                                use_pretraining=use_pretraining, pretrained_weights_path=pretrained_weights_path,
                                optimizer=optimizer, loss=loss, fine_tuning_method=fine_tuning_method)

        if initial_epoch >= num_epochs:
            print(get_time_string() + 'Not fitting the model since initial_epoch is >= num_epochs. Returning model..')
            return model

        # Start Fine-tuning
        print(get_time_string() + 'Fitting the model..')
        for e in range(initial_epoch, num_epochs):
            print_line()
            print('Starting epoch ' + str(e))
            print_line()
            completed = 0

            for x_chunk, y_chunk in chunks(train_images, train_labels, batch_size, VGG16_ARCHITECTURE):
                print(get_time_string() + 'Fitting model for chunk of size ' + str(len(x_chunk)) + '...')
                model.fit(x_chunk, y_chunk,
                          batch_size=batch_size,
                          nb_epoch=1,
                          verbose=1
                          )
                completed += len(x_chunk)
                print(get_time_string() + str(completed) + ' of ' + str(len(train_images)) + ' complete. ')

            print(get_time_string() + 'Epoch ' + str(e) + ' complete.')

            file_name = self.getCheckpointFileName(base_model_save_path=model_save_path, epoch=e)
            print(get_time_string() + 'Saving model to ' + file_name)
            model.save(file_name)

            print(get_time_string() + 'Evaluating on validation set..')
            evaluateYearbookFromModel(model=model, architecture=VGG16_ARCHITECTURE, sample=sample)

            print_line()

        # model.fit(processed_train_images, train_labels,
        #           batch_size=batch_size,
        #           nb_epoch=num_epochs,
        #           shuffle=True,
        #           verbose=1, validation_data=(processed_valid_images, valid_labels),
        #           callbacks=[self.getCheckpointer(model_save_path)],
        #           initial_epoch=initial_epoch
        #           )

        print(get_time_string() + 'Fitting complete. Returning model..')

        if model_save_path is not None:
            print(get_time_string() + 'Saving final model to ' + model_save_path + '..')
            model.save(model_save_path)

        return model

    def getVGG19(self, train_images, train_labels, load_saved_model,
                 model_save_path, use_pretraining, pretrained_weights_path, train_dir,
                 val_dir, fine_tuning_method, batch_size, num_epochs, optimizer, loss, initial_epoch, sample):
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

        print(get_time_string() + 'Creating VGG19 model..')

        img_rows, img_cols = 224, 224  # Resolution of inputs
        channels = 3

        if load_saved_model:
            if model_save_path is None:
                raise Exception('Unable to load trained model as model_save_path is None!')
            print(get_time_string() + 'Loading saved model from ' + model_save_path + '..')
            model = load_model(model_save_path)
        else:
            model = vgg19_model(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=NUM_CLASSES_YEARBOOK,
                                use_pretraining=use_pretraining, pretrained_weights_path=pretrained_weights_path,
                                optimizer=optimizer, loss=loss, fine_tuning_method=fine_tuning_method)

        if initial_epoch >= num_epochs:
            print(get_time_string() + 'Not fitting the model since initial_epoch is >= num_epochs. Returning model..')
            return model

        # Start Fine-tuning
        print(get_time_string() + 'Fitting the model..')
        for e in range(initial_epoch, num_epochs):
            print_line()
            print('Starting epoch ' + str(e))
            print_line()
            completed = 0

            for x_chunk, y_chunk in chunks(train_images, train_labels, batch_size, VGG16_ARCHITECTURE):
                print(get_time_string() + 'Fitting model for chunk of size ' + str(len(x_chunk)) + '...')
                model.fit(x_chunk, y_chunk,
                          batch_size=batch_size,
                          nb_epoch=1,
                          verbose=1
                          )
                completed += len(x_chunk)
                print(get_time_string() + str(completed) + ' of ' + str(len(train_images)) + ' complete. ')

            print(get_time_string() + 'Epoch ' + str(e) + ' complete.')

            file_name = self.getCheckpointFileName(base_model_save_path=model_save_path, epoch=e)
            print(get_time_string() + 'Saving model to ' + file_name)
            model.save(file_name)

            print(get_time_string() + 'Evaluating on validation set..')
            evaluateYearbookFromModel(model=model, architecture=VGG16_ARCHITECTURE, sample=sample)

            print_line()

        print(get_time_string() + 'Fitting complete. Returning model..')

        if model_save_path is not None:
            print(get_time_string() + 'Saving final model to ' + model_save_path + '..')
            model.save(model_save_path)

        return model

    def getResNet152(self, train_images, train_labels, load_saved_model,
                     model_save_path, use_pretraining, pretrained_weights_path, train_dir,
                     val_dir, fine_tuning_method, batch_size, num_epochs, optimizer, loss, initial_epoch, sample):
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

        if load_saved_model:
            if model_save_path is None:
                raise Exception('Unable to load trained model as model_save_path is None!')
            print(get_time_string() + 'Loading saved model from ' + model_save_path + '..')
            model = load_model(model_save_path, custom_objects={'Scale': customlayers.Scale})
        else:
            model = resnet152_model(img_rows, img_cols, channels, NUM_CLASSES_YEARBOOK, use_pretraining, pretrained_weights_path,
                                    fine_tuning_method, optimizer, loss)

        if initial_epoch >= num_epochs:
            print(get_time_string() + 'Not fitting the model since initial_epoch is >= num_epochs. Returning model..')
            return model

        # Start Fine-tuning
        print(get_time_string() + 'Fitting the model..')
        for e in range(initial_epoch, num_epochs):
            print_line()
            print('Starting epoch ' + str(e))
            print_line()
            completed = 0

            for x_chunk, y_chunk in chunks(train_images, train_labels, batch_size, RESNET152_ARCHITECTURE):
                print(get_time_string() + 'Fitting model for chunk of size ' + str(len(x_chunk)) + '...')
                model.fit(x_chunk, y_chunk,
                          batch_size=batch_size,
                          nb_epoch=1,
                          verbose=1
                          )
                completed += len(x_chunk)
                print(get_time_string() + str(completed) + ' of ' + str(len(train_images)) + ' complete. ')

            file_name = self.getCheckpointFileName(base_model_save_path=model_save_path, epoch=e)
            print(get_time_string() + 'Saving model to ' + file_name)
            model.save(file_name)

            print(get_time_string() + 'Epoch ' + str(e) + ' complete. Evaluating on validation set..')
            evaluateYearbookFromModel(model=model, architecture=RESNET152_ARCHITECTURE, sample=sample)

            print_line()

        print(get_time_string() + 'Fitting complete. Returning model..')

        if model_save_path is not None:
            print(get_time_string() + 'Saving final model to ' + model_save_path + '..')
            model.save(model_save_path)

        return model

    def getResNet50(self, train_images, train_labels, load_saved_model,
                    model_save_path, use_pretraining, pretrained_weights_path, train_dir,
                    val_dir, fine_tuning_method, batch_size, num_epochs, optimizer, loss, initial_epoch, sample):
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
        print(get_time_string() + 'Creating ResNet50 model..')

        img_rows, img_cols = 224, 224  # Resolution of inputs
        channels = 3

        if load_saved_model:
            if model_save_path is None:
                raise Exception('Unable to load trained model as model_save_path is None!')
            print(get_time_string() + 'Loading saved model from ' + model_save_path + '..')
            model = load_model(model_save_path)
        else:
            model = resnet50_model(img_rows, img_cols, channels, NUM_CLASSES_YEARBOOK, use_pretraining, pretrained_weights_path,
                                   fine_tuning_method, optimizer, loss)

        if initial_epoch >= num_epochs:
            print(get_time_string() + 'Not fitting the model since initial_epoch is >= num_epochs. Returning model..')
            return model

        # Start Fine-tuning
        print(get_time_string() + 'Fitting the model..')
        for e in range(initial_epoch, num_epochs):
            print_line()
            print('Starting epoch ' + str(e))
            print_line()
            completed = 0

            for x_chunk, y_chunk in chunks(train_images, train_labels, batch_size, RESNET50_ARCHITECTURE):
                print(get_time_string() + 'Fitting model for chunk of size ' + str(len(x_chunk)) + '...')
                model.fit(x_chunk, y_chunk,
                          batch_size=batch_size,
                          nb_epoch=1,
                          verbose=1
                          )
                completed += len(x_chunk)
                print(get_time_string() + str(completed) + ' of ' + str(len(train_images)) + ' complete. ')

            file_name = self.getCheckpointFileName(base_model_save_path=model_save_path, epoch=e)
            print(get_time_string() + 'Saving model to ' + file_name)
            model.save(file_name)

            if (e % 5 == 0):
                print(get_time_string() + 'Epoch ' + str(e) + ' complete. Evaluating on validation set..')
                evaluateYearbookFromModel(model=model, architecture=RESNET50_ARCHITECTURE, sample=sample)

            print_line()

        print(get_time_string() + 'Fitting complete. Returning model..')

        if model_save_path is not None:
            print(get_time_string() + 'Saving final model to ' + model_save_path + '..')
            model.save(model_save_path)

        return model

    def getDenseNet169(self, train_images, train_labels, load_saved_model,
                       model_save_path, use_pretraining, pretrained_weights_path, train_dir,
                       val_dir, fine_tuning_method, batch_size, num_epochs, optimizer, loss, initial_epoch, sample):
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
            model = load_model(model_save_path, custom_objects={'Scale': customlayers.Scale})
        else:
            model = densenet169_model(img_rows=img_rows, img_cols=img_cols, channels=channels,
                                      num_classes=NUM_CLASSES_YEARBOOK, use_pretraining=use_pretraining,
                                      pretrained_weights_path=pretrained_weights_path,
                                      optimizer=optimizer, loss=loss,
                                      fine_tuning_method=fine_tuning_method)

        if initial_epoch >= num_epochs:
            print(get_time_string() + 'Not fitting the model since initial_epoch is >= num_epochs. Returning model..')
            return model

        # Start Fine-tuning
        print(get_time_string() + 'Fitting the model..')
        for e in range(initial_epoch, num_epochs):
            print_line()
            print('Starting epoch ' + str(e))
            print_line()
            completed = 0

            for x_chunk, y_chunk in chunks(train_images, train_labels, batch_size, DENSENET169_ARCHITECTURE):
                print(get_time_string() + 'Fitting model for chunk of size ' + str(len(x_chunk)) + '...')
                model.fit(x_chunk, y_chunk,
                          batch_size=batch_size,
                          nb_epoch=1,
                          verbose=1
                          )
                completed += len(x_chunk)
                print(get_time_string() + str(completed) + ' of ' + str(len(train_images)) + ' complete. ')

            file_name = self.getCheckpointFileName(base_model_save_path=model_save_path, epoch=e)
            print(get_time_string() + 'Saving model to ' + file_name)
            model.save(file_name)

            print(get_time_string() + 'Epoch ' + str(e) + ' complete. Evaluating on validation set..')
            evaluateYearbookFromModel(model=model, architecture=DENSENET169_ARCHITECTURE, sample=sample)

            print_line()

        # model.fit(processed_train_images, train_labels,
        #           batch_size=batch_size,
        #           nb_epoch=num_epochs,
        #           shuffle=True,
        #           verbose=1, validation_data=(processed_valid_images, valid_labels),
        #           callbacks=[self.getCheckpointer(model_save_path)],
        #           initial_epoch=initial_epoch
        #           )

        print(get_time_string() + 'Fitting complete. Returning model..')

        if model_save_path is not None:
            print(get_time_string() + 'Saving final model to ' + model_save_path + '..')
            model.save(model_save_path)

        return model

    def getDenseNet161(self, train_images, train_labels, load_saved_model,
                       model_save_path, use_pretraining, pretrained_weights_path, train_dir,
                       val_dir, fine_tuning_method, batch_size, num_epochs, optimizer, loss, initial_epoch, sample):
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

        print(get_time_string() + 'Creating DenseNet161 model..')

        img_rows, img_cols = 224, 224  # Resolution of inputs
        channels = 3

        if load_saved_model:
            if model_save_path is None:
                raise Exception('Unable to load trained model as model_save_path is None!')
            print(get_time_string() + 'Loading saved model from ' + model_save_path + '..')
            model = load_model(model_save_path, custom_objects={'Scale': customlayers.Scale})
        else:
            model = densenet161_model(img_rows=img_rows, img_cols=img_cols, channels=channels,
                                      num_classes=NUM_CLASSES_YEARBOOK, use_pretraining=use_pretraining,
                                      pretrained_weights_path=pretrained_weights_path,
                                      optimizer=optimizer, loss=loss,
                                      fine_tuning_method=fine_tuning_method)

        if initial_epoch >= num_epochs:
            print(get_time_string() + 'Not fitting the model since initial_epoch is >= num_epochs. Returning model..')
            return model

        # Start Fine-tuning
        print(get_time_string() + 'Fitting the model..')
        for e in range(initial_epoch, num_epochs):
            print_line()
            print('Starting epoch ' + str(e))
            print_line()
            completed = 0

            for x_chunk, y_chunk in chunks(train_images, train_labels, batch_size, DENSENET161_ARCHITECTURE):
                print(get_time_string() + 'Fitting model for chunk of size ' + str(len(x_chunk)) + '...')
                model.fit(x_chunk, y_chunk,
                          batch_size=batch_size,
                          nb_epoch=1,
                          verbose=1
                          )
                completed += len(x_chunk)
                print(get_time_string() + str(completed) + ' of ' + str(len(train_images)) + ' complete. ')

            file_name = self.getCheckpointFileName(base_model_save_path=model_save_path, epoch=e)
            print(get_time_string() + 'Saving model to ' + file_name)
            model.save(file_name)

            print(get_time_string() + 'Epoch ' + str(e) + ' complete. Evaluating on validation set..')
            evaluateYearbookFromModel(model=model, architecture=DENSENET161_ARCHITECTURE, sample=sample)

            print_line()

        # model.fit(processed_train_images, train_labels,
        #           batch_size=batch_size,
        #           nb_epoch=num_epochs,
        #           shuffle=True,
        #           verbose=1, validation_data=(processed_valid_images, valid_labels),
        #           callbacks=[self.getCheckpointer(model_save_path)],
        #           initial_epoch=initial_epoch
        #           )

        print(get_time_string() + 'Fitting complete. Returning model..')

        if model_save_path is not None:
            print(get_time_string() + 'Saving final model to ' + model_save_path + '..')
            model.save(model_save_path)

        return model

    def getDenseNet121(self, train_images, train_labels, load_saved_model,
                       model_save_path, use_pretraining, pretrained_weights_path, train_dir,
                       val_dir, fine_tuning_method, batch_size, num_epochs, optimizer, loss, initial_epoch, sample):
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

        print(get_time_string() + 'Creating DenseNet121 model..')

        img_rows, img_cols = 224, 224  # Resolution of inputs
        channels = 3

        if load_saved_model:
            if model_save_path is None:
                raise Exception('Unable to load trained model as model_save_path is None!')
            print(get_time_string() + 'Loading saved model from ' + model_save_path + '..')
            model = load_model(model_save_path, custom_objects={'Scale': customlayers.Scale})
        else:
            model = densenet121_model(img_rows=img_rows, img_cols=img_cols, channels=channels,
                                      num_classes=NUM_CLASSES_YEARBOOK, use_pretraining=use_pretraining,
                                      pretrained_weights_path=pretrained_weights_path,
                                      optimizer=optimizer, loss=loss,
                                      fine_tuning_method=fine_tuning_method)

        if initial_epoch >= num_epochs:
            print(get_time_string() + 'Not fitting the model since initial_epoch is >= num_epochs. Returning model..')
            return model

        # Start Fine-tuning
        print(get_time_string() + 'Fitting the model..')
        for e in range(initial_epoch, num_epochs):
            print_line()
            print('Starting epoch ' + str(e))
            print_line()
            completed = 0

            for x_chunk, y_chunk in chunks(train_images, train_labels, batch_size, DENSENET121_ARCHITECTURE):
                print(get_time_string() + 'Fitting model for chunk of size ' + str(len(x_chunk)) + '...')
                model.fit(x_chunk, y_chunk,
                          batch_size=batch_size,
                          nb_epoch=1,
                          verbose=1
                          )
                completed += len(x_chunk)
                print(get_time_string() + str(completed) + ' of ' + str(len(train_images)) + ' complete. ')

            file_name = self.getCheckpointFileName(base_model_save_path=model_save_path, epoch=e)
            print(get_time_string() + 'Saving model to ' + file_name)
            model.save(file_name)

            print(get_time_string() + 'Epoch ' + str(e) + ' complete. Evaluating on validation set..')
            evaluateYearbookFromModel(model=model, architecture=DENSENET121_ARCHITECTURE, sample=sample)

            print_line()

        # model.fit(processed_train_images, train_labels,
        #           batch_size=batch_size,
        #           nb_epoch=num_epochs,
        #           shuffle=True,
        #           verbose=1, validation_data=(processed_valid_images, valid_labels),
        #           callbacks=[self.getCheckpointer(model_save_path)],
        #           initial_epoch=initial_epoch
        #           )

        print(get_time_string() + 'Fitting complete. Returning model..')

        if model_save_path is not None:
            print(get_time_string() + 'Saving final model to ' + model_save_path + '..')
            model.save(model_save_path)

        return model
