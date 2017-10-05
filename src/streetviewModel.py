from util import *
from kagglenet import *
from keras.models import load_model

class StreetViewModel:
    get_model_function = {}

    def __init__(self):
        self.get_model_function[KAGGLE_ARCHITECTURE] = self.getKaggleModel

    def getModel(self, model_architecture='kaggle', load_saved_model=False, model_save_path=None,
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
        train_data = listStreetView(True, False, sample)

        train_images, train_gps = get_streetview_data_and_labels(train_data, STREETVIEW_TRAIN_PATH)

        return self.get_model_function[model_architecture](train_images, train_gps,
                                                           load_saved_model,
                                                           model_save_path,
                                                           use_pretraining,
                                                           pretrained_weights_path,
                                                           fine_tuning_method,
                                                           batch_size, num_epochs,
                                                           optimizer, loss,
                                                           initial_epoch,
                                                           sample)

    def getKaggleModel(self, train_images, train_gps, load_saved_model,
                       model_save_path, use_pretraining, pretrained_weights_path,
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

        print(get_time_string() + 'Creating Kaggle model..')

        img_rows, img_cols = 224, 224  # Resolution of inputs
        channels = 3

        if load_saved_model:
            if model_save_path is None:
                raise Exception('Unable to load trained model as model_save_path is None!')
            print(get_time_string() + 'Loading saved model from ' + model_save_path + '..')
            model = load_model(model_save_path)
        else:
            model = KaggleNetModel(img_rows=img_rows, img_cols=img_cols, channels=channels, num_classes=2,
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

            for x_chunk, y_chunk in chunks(train_images, train_gps, batch_size, KAGGLE_ARCHITECTURE):
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
            evaluateStreetviewFromModel(model=model, architecture=KAGGLE_ARCHITECTURE, sample=sample)

            print_line()

        print(get_time_string() + 'Fitting complete. Returning model..')

        if model_save_path is not None:
            print(get_time_string() + 'Saving final model to ' + model_save_path + '..')
            model.save(model_save_path)

        return model

    def getCheckpointFileName(self, base_model_save_path, epoch):
        ext = '.h5'
        path_wo_ext = base_model_save_path.split(ext)[0]
        filepath = path_wo_ext + '-' + str(epoch) + ext
        return filepath


