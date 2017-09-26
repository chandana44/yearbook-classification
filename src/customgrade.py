from __future__ import print_function

from math import sin, cos, atan2, sqrt, pi

from model import *
from run import *

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

ALEXNET_ARCHITECTURE = 'alexnet'
CHECKPOINT_BASE_DIR = '../checkpoint/'
ALEXNET_PRETRAINED_WEIGHT_PATH = '../pretrained_weights/alexnet_weights.h5'


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
def evaluateYearbookFromModel(model):
    val_list = util.listYearbook(False, True)
    predictor = Predictor()
    predictor.DATASET_TYPE = 'yearbook'

    total_count = len(val_list)
    l1_dist = 0.0
    print("Total validation data: ", total_count)
    for ground_truth_entry in val_list:
        full_file_path = path.join(YEARBOOK_VALID_PATH, ground_truth_entry[0])
        pred_year = np.argmax(model.predict(preprocess_image_batch([full_file_path],
                                                                   img_size=(256, 256),
                                                                   crop_size=(227, 227),
                                                                   color_mode="rgb"))) + 1900
        truth_year = int(ground_truth_entry[1])
        l1_dist += abs(pred_year - truth_year)

    l1_dist /= total_count
    print("L1 distance for validation set: ", l1_dist)
    return l1_dist


# Evaluate L1 distance on valid data for yearbook dataset
def evaluateYearbook(Predictor):
    test_list = util.listYearbook(False, True)
    predictor = Predictor()
    predictor.DATASET_TYPE = 'yearbook'

    total_count = len(test_list)
    l1_dist = 0.0
    print("Total validation data", total_count)
    count = 0
    for image_gr_truth in test_list:
        image_path = path.join(YEARBOOK_VALID_PATH, image_gr_truth[0])
        pred_year = predictor.predict(image_path)
        truth_year = int(image_gr_truth[1])
        l1_dist += abs(pred_year[0] - truth_year)
        count += 1

    l1_dist /= total_count
    print("L1 distance", l1_dist)
    return l1_dist


# Evaluate L1 distance on valid data for geolocation dataset
def evaluateStreetview(Predictor):
    test_list = listStreetView(False, True)
    predictor = Predictor()
    predictor.DATASET_TYPE = 'geolocation'

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
def predictTestYearbookFromModel(model):
    test_list = util.testListYearbook()
    predictor = Predictor()
    predictor.DATASET_TYPE = 'yearbook'

    total_count = len(test_list)
    print("Total test data: ", total_count)

    test_images = [path.join(YEARBOOK_TEST_PATH, item[0]) for item in test_list]
    processed_test_images = preprocess_image_batch(test_images, img_size=(256, 256), crop_size=(227, 227),
                                                   color_mode="rgb")

    output = open(YEARBOOK_TEST_LABEL_PATH, 'w')
    for image in processed_test_images:
        pred_year = np.argmax(model.predict(np.stack([image], axis=0))) + 1900
        out_string = str(pred_year) + '\n'
        output.write(out_string)
    output.close()


# Predict label for test data on yearbook dataset
def predictTestYearbook(Predictor):
    test_list = util.testListYearbook()
    predictor = Predictor()
    predictor.DATASET_TYPE = 'yearbook'

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
def predictTestStreetview(Predictor):
    test_list = testListStreetView()
    predictor = Predictor()
    predictor.DATASET_TYPE = 'geolocation'

    total_count = len(test_list)
    print("Total test data", total_count)

    output = open(STREETVIEW_TEST_LABEL_PATH, 'w')
    for image in test_list:
        image_path = path.join(STREETVIEW_TEST_PATH, image[0])
        pred_lat, pred_lon = predictor.predict(image_path)
        out_string = image[0] + '\t' + str(pred_lat) + '\t' + str(pred_lon) + '\n'
        output.write(out_string)
    output.close()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser("Evaluate a model on the validation set")
    parser.add_argument("--DATASET_TYPE", dest="dataset_type",
                        help="Dataset: yearbook/geolocation", required=True)
    parser.add_argument("--type", dest="type",
                        help="Dataset: valid/test", required=True)

    args = parser.parse_args()
    print(get_time_string() + 'Operating on ' + args.dataset_type + ' dataset..')

    if args.dataset_type == 'yearbook':
        model = YearbookModel()
        trained_model = model.getModel(model_architecture=ALEXNET_ARCHITECTURE, load_saved_model=False,
                                       model_save_path=CHECKPOINT_BASE_DIR + 'alexnet_saved_model1.h5',
                                       use_pretraining=True,
                                       pretrained_weights_path=ALEXNET_PRETRAINED_WEIGHT_PATH,
                                       train_dir=None, val_dir=None, fine_tuning_method=None)
        if args.type == 'valid':
            # evaluateYearbook(Predictor)
            evaluateYearbookFromModel(trained_model)
        elif args.type == 'test':
            # predictTestYearbook(Predictor)
            predictTestYearbookFromModel(trained_model)
        else:
            print(get_time_string() + "Unknown type '%s'", args.type)
    elif args.dataset_type == 'geolocation':
        if args.type == 'valid':
            evaluateStreetview(Predictor)
        elif args.type == 'test':
            predictTestStreetview(Predictor)
        else:
            print(get_time_string() + "Unknown type '%s'", args.type)
    else:
        print(get_time_string() + "Unknown dataset type '%s'", Predictor.DATASET_TYPE)
        exit(1)
