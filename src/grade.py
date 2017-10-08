from __future__ import print_function
from os import path
from math import sin, cos, atan2, sqrt, pi
from run import *
from util import *
from model import *
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

CHECKPOINT_BASE_DIR = '../model/'
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


def numToRadians(x):
    return x / 180.0 * pi


# Calculate L1 score: max, min, mean, standard deviation
def maxScore(ground_truth, predicted_values):
    diff = np.absolute(np.array(ground_truth) - np.array(predicted_values))
    diff_sum = np.sum(diff, axis=1)
    return np.max(diff_sum), np.min(diff_sum), np.mean(diff_sum), np.std(diff_sum)


# Calculate distance (km) between Latitude/Longitude points
# Reference: http://www.movable-type.co.uk/scripts/latlong.html
EARTH_RADIUS = 6371


def dist(lat1, lon1, lat2, lon2):
    lat1 = numToRadians(lat1)
    lon1 = numToRadians(lon1)
    lat2 = numToRadians(lat2)
    lon2 = numToRadians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2.0) * sin(dlat / 2.0) + cos(lat1) * cos(lat2) * sin(dlon / 2.0) * sin(dlon / 2.0)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    d = EARTH_RADIUS * c
    return d


# Evaluate L1 distance on valid data for yearbook dataset
def evaluateYearbook(predictor):
    test_list = util.listYearbook(False, True)

    total_count = len(test_list)
    l1_dist = 0.0
    print("Total validation data", total_count)
    count = 0
    for image_gr_truth in test_list:
        image_path = path.join(YEARBOOK_VALID_PATH, image_gr_truth[0])
        pred_year = predictor.predict(image_path)
        truth_year = int(image_gr_truth[1])
        l1_dist += abs(pred_year[0] - truth_year)
        count = count + 1

    l1_dist /= total_count
    print("L1 distance", l1_dist)
    return l1_dist


# Evaluate L1 distance on valid data for geolocation dataset
def evaluateStreetview(predictor):
    test_list = listStreetView(False, True)

    total_count = len(test_list)
    l1_dist = 0
    print("Total validation data", total_count)
    for image_gr_truth in test_list:
        image_path = path.join(STREETVIEW_VALID_PATH, image_gr_truth[0])
        pred_lat, pred_lon = predictor.predict(image_path)
        truth_lat, truth_lon = float(image_gr_truth[1]), float(image_gr_truth[2])
        l1_dist += dist(pred_lat, pred_lon, truth_lat, truth_lon)
    l1_dist /= total_count
    print("L1 distance", l1_dist)
    return l1_dist


# Predict label for test data on yearbook dataset
def predictTestYearbook(predictor):
    test_list = util.testListYearbook()

    total_count = len(test_list)
    print("Total test data: ", total_count)

    output = open(YEARBOOK_TEST_LABEL_PATH, 'w')
    for image in test_list:
        image_path = path.join(YEARBOOK_TEST_PATH, image[0])
        pred_year = predictor.predict(image_path)
        out_string = image[0] + '\t' + str(pred_year[0]) + '\n'
        output.write(out_string)
    output.close()


# Predict label for test data for geolocation dataset
def predictTestStreetview(predictor):
    test_list = testListStreetView()

    total_count = len(test_list)
    print("Total test data", total_count)

    output = open(STREETVIEW_TEST_LABEL_PATH, 'w')
    for image in test_list:
        image_path = path.join(STREETVIEW_TEST_PATH, image[0])
        pred_lat, pred_lon = predictor.predict(image_path)
        out_string = image[0] + '\t' + str(pred_lat) + '\t' + str(pred_lon) + '\n'
        output.write(out_string)
    output.close()


def getModels(dataset, models_checkpoints, use_pretraining=True,
              pretrained_weights_path=None,
              train_dir=None, val_dir=None, fine_tuning_method=None,
              batch_size=None, num_epochs=10,
              optimizer='sgd', loss='mse',
              initial_epoch=0,
              sample=0, width=10, height=10):
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


if __name__ == "__main__":
    import importlib
    from argparse import ArgumentParser

    parser = ArgumentParser("Evaluate a model on the validation set")
    parser.add_argument("--DATASET_TYPE", dest="dataset_type",
                        help="Dataset: yearbook/geolocation", required=True)
    parser.add_argument("--type", dest="type",
                        help="Dataset: valid/test", required=True)
    parser.add_argument("--ensemble_models", dest="ensemble_models",
                        help="ensemble_models: alexnet:checkpoint1.h5,vgg16:checkpoint2.h5 etc",
                        required=False, default=None)

    args = parser.parse_args()
    models_checkpoints = args.ensemble_models.split(',')
    models_architectures_tuples = getModels(dataset=args.dataset_type, models_checkpoints=models_checkpoints,
                                            pretrained_weights_path=pretrained_weights_path_map[
                                                args.model_architecture])
    predictor = Predictor(args.dataset_type, models_architectures_tuples)

    if args.dataset_type == 'yearbook':
        print("Yearbook")
        if (args.type == 'valid'):
            evaluateYearbook(predictor)
        elif (args.type == 'test'):
            predictTestYearbook(predictor)
        else:
            print("Unknown type '%s'", args.type)
    elif args.dataset_type == 'geolocation':
        print("Geolocation")
        if (args.type == 'valid'):
            evaluateStreetview(predictor)
        elif (args.type == 'test'):
            predictTestStreetview(predictor)
        else:
            print("Unknown type '%s'", args.type)
    else:
        print("Unknown dataset type '%s'", args.dataset_type)
        exit(1)
