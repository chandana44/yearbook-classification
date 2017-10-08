from __future__ import print_function

import os

os.environ['THEANO_FLAGS'] = "device=cuda0"

from argparse import ArgumentParser
from model import *
from run import *
from streetviewModel import *

SRC_PATH = path.dirname(path.abspath(__file__))
DATA_PATH = path.join(SRC_PATH, '..', 'data')
YEARBOOK_PATH = path.join(DATA_PATH, 'yearbook')
YEARBOOK_VALID_PATH = path.join(YEARBOOK_PATH, 'valid')
YEARBOOK_TEST_PATH = path.join(YEARBOOK_PATH, 'test')
YEARBOOK_TEST_LABEL_PATH = path.join(SRC_PATH, '..', 'output', 'yearbook_test_label.txt')
STREETVIEW_PATH = path.join(DATA_PATH, 'geo')
STREETVIEW_VALID_PATH = path.join(STREETVIEW_PATH, 'valid')
STREETVIEW_TEST_PATH = path.join(STREETVIEW_PATH, 'test')
STREETVIEW_TEST_LABEL_PATH = path.join(SRC_PATH, '..', 'output', 'geo_test_label.txt')

CHECKPOINT_BASE_DIR = '../checkpoint/'
ALEXNET_PRETRAINED_WEIGHT_PATH = '../pretrained_weights/alexnet_weights.h5'
VGG16_PRETRAINED_WEIGHT_PATH = '../pretrained_weights/vgg16_weights_th_dim_ordering_th_kernels.h5'
VGG19_PRETRAINED_WEIGHT_PATH = '../pretrained_weights/vgg19_weights_th_dim_ordering_th_kernels.h5'
RESNET152_PRETRAINED_WEIGHT_PATH = '../pretrained_weights/resnet152_weights.h5'
RESNET50_PRETRAINED_WEIGHT_PATH = '../pretrained_weights/resnet50_weights.h5'
DENSENET169_PRETRAINED_WEIGHT_PATH = '../pretrained_weights/densenet169_weights_th.h5'
DENSENET121_PRETRAINED_WEIGHT_PATH = '../pretrained_weights/densenet121_weights_th.h5'
DENSENET161_PRETRAINED_WEIGHT_PATH = '../pretrained_weights/densenet161_weights_th.h5'
KAGGLE_PRETRAINED_WEIGHT_PATH = ''

pretrained_weights_path_map = {ALEXNET_ARCHITECTURE: ALEXNET_PRETRAINED_WEIGHT_PATH,
                               ALEXNET_REGRESSION_ARCHITECTURE: ALEXNET_PRETRAINED_WEIGHT_PATH,
                               VGG16_ARCHITECTURE: VGG16_PRETRAINED_WEIGHT_PATH,
                               VGG19_ARCHITECTURE: VGG19_PRETRAINED_WEIGHT_PATH,
                               RESNET152_ARCHITECTURE: RESNET152_PRETRAINED_WEIGHT_PATH,
                               RESNET50_ARCHITECTURE: RESNET50_PRETRAINED_WEIGHT_PATH,
                               KERAS_RESNET50_ARCHITECTURE: '',
                               DENSENET169_ARCHITECTURE: DENSENET169_PRETRAINED_WEIGHT_PATH,
                               DENSENET161_ARCHITECTURE: DENSENET161_PRETRAINED_WEIGHT_PATH,
                               DENSENET121_ARCHITECTURE: DENSENET121_PRETRAINED_WEIGHT_PATH,
                               KAGGLE_ARCHITECTURE: KAGGLE_PRETRAINED_WEIGHT_PATH}


def getYearbookTestOutputFile(checkpoint_file):
    checkpoint_ext = '.h5'
    checkpoint_file_wo_ext = checkpoint_file.split(checkpoint_ext)[0]

    output_path = YEARBOOK_TEST_LABEL_PATH
    ext = '.txt'
    output_path_wo_ext = output_path.split(ext)[0]

    return output_path_wo_ext + '-' + checkpoint_file_wo_ext + ext


def getGeolocationTestOutputFile(checkpoint_file):
    checkpoint_ext = '.h5'
    checkpoint_file_wo_ext = checkpoint_file.split(checkpoint_ext)[0]

    output_path = STREETVIEW_TEST_LABEL_PATH
    ext = '.txt'
    output_path_wo_ext = output_path.split(ext)[0]

    return output_path_wo_ext + '-' + checkpoint_file_wo_ext + ext


# Predict label for test data on yearbook dataset
def predictTestYearbookFromModel(model, architecture, checkpoint_file, sample=False):
    test_list = util.testListYearbook(sample=sample)
    test_images = [path.join(YEARBOOK_TEST_PATH, item[0]) for item in test_list]

    total_count = len(test_list)
    print(get_time_string() + "Total test data: ", total_count)

    batch_size = 128
    count = 0

    output_file = getYearbookTestOutputFile(checkpoint_file)
    output = open(output_file, 'w')
    for x_chunk, image_name_chunk in chunks_test(test_images, test_list, batch_size, architecture):
        print(get_time_string() + 'Testing ' + str(count + 1) + ' - ' + str(count + batch_size))
        predictions = model.predict(x_chunk)
        years = np.array([np.argmax(p) + 1900 for p in predictions])
        i = 0
        for pred_year in years:
            out_string = image_name_chunk[i] + '\t' + str(pred_year) + '\n'
            output.write(out_string)
            i += 1
        count += batch_size
    output.close()


# Predict label for test data on yearbook dataset
def predictTestGeoLocationFromModel(model, architecture, checkpoint_file, width, height, sample=False):
    min_x, max_x, min_y, max_y = get_min_max_xy_geo()

    test_list = util.testListStreetView(sample=sample)
    test_images = [path.join(YEARBOOK_TEST_PATH, item[0]) for item in test_list]

    total_count = len(test_list)
    print(get_time_string() + "Total test data: ", total_count)

    batch_size = 128
    count = 0

    output_file = getGeolocationTestOutputFile(checkpoint_file)
    output = open(output_file, 'w')
    for x_chunk, image_name_chunk in chunks_test(test_images, test_list, batch_size, architecture):
        print(get_time_string() + 'Testing ' + str(count + 1) + ' - ' + str(count + batch_size))
        batch_len = len(x_chunk)
        predictions = model.predict(x_chunk)

        if architecture in CLASSIFICATION_MODELS:  # Classification nets
            int_labels = np.array([np.argmax(p) for p in predictions])
            xy_coordinates = np.zeros((batch_len, 2))
            for i in range(batch_len):
                int_label = int_labels[i]
                x, y = get_xy_from_int_label(width, height, int_label, min_x, max_x, min_y, max_y)
                xy_coordinates[i, 0] = x
                xy_coordinates[i, 1] = y
            coordinates = XYToCoordinate(xy_coordinates)
            for i in range(batch_len):
                out_string = image_name_chunk[i] + '\t' + str(coordinates[i, 0]) + '\t' + str(coordinates[i, 1]) + '\n'
                output.write(out_string)
        else:  # Regression nets
            latslongs = np.array([[p[0], p[1]] for p in predictions])
            for i in range(batch_len):
                out_string = image_name_chunk[i] + '\t' + str(latslongs[i][0]) + '\t' + str(latslongs[i][1]) + '\n'
                output.write(out_string)

        count += batch_size
    output.close()


def getModels(dataset, models_checkpoints, use_pretraining=True,
              pretrained_weights_path=None,
              train_dir=None, val_dir=None, fine_tuning_method=None,
              batch_size=None, num_epochs=10,
              optimizer='sgd', loss='mse',
              initial_epoch=0,
              sample=0, width=0, height=0):
    models_architectures_tuples = []
    for model_checkpoint in models_checkpoints:
        architecture = model_checkpoint.split(':')[0]
        checkpoint_file_name = model_checkpoint.split(':')[1]

        if architecture not in ARCHITECTURES:
            raise Exception('Invalid architecture type!')
        if dataset == 'yearbook':
            yearbookModel = YearbookModel()
            this_model = yearbookModel.getModel(model_architecture=architecture, load_saved_model=1,
                                                model_save_path=CHECKPOINT_BASE_DIR + checkpoint_file_name,
                                                initial_epoch=initial_epoch, use_pretraining=use_pretraining,
                                                pretrained_weights_path=pretrained_weights_path,
                                                fine_tuning_method=fine_tuning_method, batch_size=batch_size,
                                                num_epochs=num_epochs, optimizer=optimizer, loss=loss,
                                                sample=sample)
        elif dataset == 'geolocation':
            streetviewModel = StreetViewModel()
            this_model = streetviewModel.getModel(model_architecture=architecture, load_saved_model=1,
                                                  model_save_path=CHECKPOINT_BASE_DIR + checkpoint_file_name,
                                                  initial_epoch=initial_epoch, use_pretraining=use_pretraining,
                                                  pretrained_weights_path=pretrained_weights_path,
                                                  fine_tuning_method=fine_tuning_method, batch_size=batch_size,
                                                  num_epochs=num_epochs, optimizer=optimizer, loss=loss,
                                                  sample=sample, width=width, height=height)
        else:
            raise Exception('Unknown dataset type')
        models_architectures_tuples.append((this_model, architecture))

    return models_architectures_tuples


def getModelsMap(individual_models_2d, use_pretraining=True,
                 pretrained_weights_path=None,
                 train_dir=None, val_dir=None, fine_tuning_method=None,
                 batch_size=None, num_epochs=10,
                 optimizer='sgd', loss='mse',
                 initial_epoch=0,
                 sample=0):
    models_map = {}
    for models_1d in individual_models_2d:
        for model_checkpoint in models_1d:
            if model_checkpoint not in models_map:  # If model not already created
                architecture = model_checkpoint.split(':')[0]
                checkpoint_file_name = model_checkpoint.split(':')[1]

                if architecture not in ARCHITECTURES:
                    raise Exception('Invalid architecture type!')

                yearbookModel = YearbookModel()
                this_model = yearbookModel.getModel(model_architecture=architecture, load_saved_model=1,
                                                    model_save_path=CHECKPOINT_BASE_DIR + checkpoint_file_name,
                                                    initial_epoch=initial_epoch, use_pretraining=use_pretraining,
                                                    pretrained_weights_path=pretrained_weights_path,
                                                    fine_tuning_method=fine_tuning_method, batch_size=batch_size,
                                                    num_epochs=num_epochs, optimizer=optimizer, loss=loss,
                                                    sample=sample)
                models_map[model_checkpoint] = this_model
    return models_map


if __name__ == "__main__":
    if is_using_gpu():
        print('Program is using GPU..')
    else:
        print('Program is using CPU..')

    parser = ArgumentParser("Evaluate a model on the validation set")
    parser.add_argument("--DATASET_TYPE", dest="dataset_type",
                        help="Dataset: yearbook/geolocation", required=True)
    parser.add_argument("--type", dest="type",
                        help="Dataset: valid/test", required=True)

    parser.add_argument("--model_architecture", dest="model_architecture",
                        help="Model architecture: alexnet/vgg16/vgg19/resnet152/resnet50/densenet169/densenet121/densenet161",
                        required=True)
    parser.add_argument("--load_saved_model", dest="load_saved_model",
                        help="load_saved_model: Whether to use saved model",
                        required=False, default=0, type=int)
    parser.add_argument("--initial_epoch", dest="initial_epoch",
                        help="initial_epoch: Epoch to start training from",
                        required=False, default=0, type=int)
    parser.add_argument("--checkpoint_file_name", dest="checkpoint_file_name",
                        help="checkpoint_file_name: h5 file name to save to/load from", required=True)

    parser.add_argument("--use_pretraining", dest="use_pretraining",
                        help="use_pretraining: Whether to use supervised pretraining",
                        required=False, default=1, type=int)
    parser.add_argument("--fine_tuning_method", dest="fine_tuning_method",
                        help="fine_tuning_method: end-to-end/phase-by-phase",
                        required=False)

    parser.add_argument("--batch_size", dest="batch_size",
                        help="batch_size: size of batches while fitting the model",
                        required=False, default=128, type=int)
    parser.add_argument("--num_epochs", dest="num_epochs",
                        help="num_epochs: number of epochs to train the model for",
                        required=False, default=20, type=int)
    parser.add_argument("--loss", dest="loss",
                        help="loss function to use: l1|mse|mae",
                        required=True)
    parser.add_argument("--optimizer", dest="optimizer",
                        help="optimizer to use: sgd|adagrad",
                        required=False, default='sgd')
    parser.add_argument("--lr", dest="lr",
                        help="learning rate: 0.001",
                        required=False, default=None, type=float)

    parser.add_argument("--sample", dest="sample",
                        help="sample: whether to use sample dataset",
                        required=False, default=0, type=int)

    parser.add_argument("--ensemble", dest="ensemble",
                        help="ensemble: whether to use multiple saved models to calculate scores on validation/test "
                             "data",
                        required=False, default=0, type=int)
    parser.add_argument("--ensemble_models", dest="ensemble_models",
                        help="ensemble_models: alexnet:checkpoint1.h5,vgg16:checkpoint2.h5 etc",
                        required=False, default=None)

    parser.add_argument("--width", dest="width",
                        help="width: number of classes to partition the x-coordinate",
                        required=False, default=20, type=int)
    parser.add_argument("--height", dest="height",
                        help="height: number of classes to partition the y-coordinate",
                        required=False, default=20, type=int)

    args = parser.parse_args()
    print('Args provided: ' + str(args))

    print(get_time_string() + 'Operating on ' + args.dataset_type + ' dataset..')

    if args.ensemble == 1:
        if args.ensemble_models is None:
            raise Exception('ensemble is 1 but no models/checkpoint files specified!')
        # models_checkpoints = args.ensemble_models.split(
        #     ',')  # array of entries of format <architecture>:<checkpoint_file>
        # models_architectures_tuples = getModels(models_checkpoints, use_pretraining=args.use_pretraining,
        #                                         pretrained_weights_path=pretrained_weights_path_map[args.model_architecture],
        #                                         train_dir=None, val_dir=None,
        #                                         fine_tuning_method=args.fine_tuning_method,
        #                                         batch_size=args.batch_size, num_epochs=args.num_epochs,
        #                                         optimizer=args.optimizer, loss=args.loss,
        #                                         initial_epoch=1000,
        #                                         # Just returning the model without any further training
        #                                         sample=args.sample)

        models_architectures_tuples_list = []
        ensembled_models = args.ensemble_models.split(
            '#')  # array of entires of format <model_checkpoints1>#<model_checkpoints2>

        individual_models_2d = []
        for ensembled_model in ensembled_models:
            models_checkpoints = ensembled_model.split(
                ',')  # array of entries of format <architecture>:<checkpoint_file>
            individual_models_2d.append(models_checkpoints)

        print(str(individual_models_2d))
        models_map = getModelsMap(individual_models_2d, use_pretraining=args.use_pretraining,
                                  pretrained_weights_path=pretrained_weights_path_map[args.model_architecture],
                                  train_dir=None, val_dir=None,
                                  fine_tuning_method=args.fine_tuning_method,
                                  batch_size=args.batch_size, num_epochs=args.num_epochs,
                                  optimizer=args.optimizer, loss=args.loss,
                                  initial_epoch=1000,
                                  # Just returning the model without any further training
                                  sample=args.sample)

        if args.type == 'valid':
            evaluateYearbookFromEnsembledModelsMultiple(models_map, individual_models_2d,
                                                        sample=args.sample)
            # evaluateFromEnsembledModels(args.dataset_type, models_architectures_tuples=models_architectures_tuples,
            #                             sample=args.sample, width=args.width, height=args.height)
        elif args.type == 'test':  # TODO implement ensembling while testing also
            pass
            # predictTestYearbookFromModel(trained_model, args.model_architecture, args.sample)
        else:
            print(get_time_string() + "Unknown type '%s'", args.type)

        exit(0)

    if args.model_architecture not in ARCHITECTURES:
        raise Exception('Invalid model architecture type!')

    if args.fine_tuning_method is not None and args.fine_tuning_method not in FINE_TUNING_METHODS:
        raise Exception('Invalid fine_tuning_method specified!')

    if not args.checkpoint_file_name.endswith('.h5'):
        raise Exception('Checkpoint file should end with h5 format!')

    if args.dataset_type == 'yearbook':
        model = YearbookModel()
        trained_model = model.getModel(model_architecture=args.model_architecture,
                                       load_saved_model=args.load_saved_model,
                                       model_save_path=CHECKPOINT_BASE_DIR + args.checkpoint_file_name,
                                       use_pretraining=args.use_pretraining,
                                       pretrained_weights_path=pretrained_weights_path_map[args.model_architecture],
                                       train_dir=None, val_dir=None, fine_tuning_method=args.fine_tuning_method,
                                       batch_size=args.batch_size, num_epochs=args.num_epochs,
                                       optimizer=args.optimizer, loss=args.loss,
                                       initial_epoch=args.initial_epoch,
                                       sample=args.sample, lr=args.lr)
        if args.type == 'valid':
            evaluateYearbookFromModel(trained_model, args.model_architecture, args.sample)
        elif args.type == 'test':
            predictTestYearbookFromModel(trained_model, args.model_architecture, args.checkpoint_file_name, args.sample)
        else:
            print(get_time_string() + "Unknown type '%s'", args.type)
    elif args.dataset_type == 'geolocation':
        model = StreetViewModel()
        trained_model = model.getModel(model_architecture=args.model_architecture,
                                       load_saved_model=args.load_saved_model,
                                       model_save_path=CHECKPOINT_BASE_DIR + args.checkpoint_file_name,
                                       use_pretraining=args.use_pretraining,
                                       pretrained_weights_path=pretrained_weights_path_map[args.model_architecture],
                                       train_dir=None, val_dir=None, fine_tuning_method=args.fine_tuning_method,
                                       batch_size=args.batch_size, num_epochs=args.num_epochs,
                                       optimizer=args.optimizer, loss=args.loss,
                                       initial_epoch=args.initial_epoch,
                                       sample=args.sample, width=args.width,
                                       height=args.height, lr=args.lr)
        if args.type == 'valid':
            if args.model_architecture in CLASSIFICATION_MODELS:  # Classification nets
                evaluateStreetviewFromModelClassification(trained_model, args.model_architecture, width=args.width,
                                                          height=args.height, sample=args.sample)
            else:  # Regression nets
                evaluateStreetviewFromModel(trained_model, args.model_architecture, args.sample)
        elif args.type == 'test':
            predictTestGeoLocationFromModel(trained_model, args.model_architecture, args.checkpoint_file_name,
                                            args.widgth, args.height, args.sample)
        else:
            print(get_time_string() + "Unknown type '%s'", args.type)
    else:
        print(get_time_string() + "Enter yearbook/geolocation for dataset")
        exit(1)
