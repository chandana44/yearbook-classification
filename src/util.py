# Feel free to modify this to your needs. We will not rely on your util.py
import re
import time
from os import path
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from theano import function, config, shared, tensor
from keras import backend as K
from math import sin, cos, atan2, sqrt, pi, ceil, floor

# If you want this to work do not move this file
SRC_PATH = path.dirname(path.abspath(__file__))
DATA_PATH = path.join(SRC_PATH, '..', 'data')

YEARBOOK_PATH = path.join(DATA_PATH, "yearbook")
YEARBOOK_TXT_PREFIX = path.join(YEARBOOK_PATH, "yearbook")
YEARBOOK_TXT_SAMPLE_PREFIX = path.join(YEARBOOK_PATH, "yearbook_sample")

YEARBOOK_TRAIN_PATH = path.join(YEARBOOK_PATH, 'train')
YEARBOOK_VALID_PATH = path.join(YEARBOOK_PATH, 'valid')
YEARBOOK_TEST_PATH = path.join(YEARBOOK_PATH, 'test')
YEARBOOK_TEST_LABEL_PATH = path.join(SRC_PATH, '..', 'output', 'yearbook_test_label.txt')
YEARBOOK_TEST_LABEL_PREFIX = path.join(SRC_PATH, '..', 'output', 'yearbook_test_')

STREETVIEW_PATH = path.join(DATA_PATH, "geo")
STREETVIEW_TXT_PREFIX = path.join(STREETVIEW_PATH, "geo")
STREETVIEW_TXT_SAMPLE_PREFIX = path.join(STREETVIEW_PATH, "geo_sample")

STREETVIEW_TRAIN_PATH = path.join(STREETVIEW_PATH, 'train')
STREETVIEW_VALID_PATH = path.join(STREETVIEW_PATH, 'valid')
STREETVIEW_TEST_PATH = path.join(STREETVIEW_PATH, 'test')
STREETVIEW_TEST_LABEL_PATH = path.join(SRC_PATH, '..', 'output', 'geo_test_label.txt')
STREETVIEW_TEST_LABEL_PREFIX = path.join(SRC_PATH, '..', 'output', 'geo_test_')

NUM_CLASSES_YEARBOOK = 118
NUM_CLASSES_GEOLOCATION = 2

yb_r = re.compile("(\d\d\d\d)_(.*)_(.*)_(.*)_(.*)")
sv_r = re.compile("([+-]?\d*\.\d*)_([+-]?\d*\.\d*)_\d*_-004")

# architectures
ALEXNET_ARCHITECTURE = 'alexnet'
ALEXNET_REGRESSION_ARCHITECTURE = 'alexnet-regression'
VGG16_ARCHITECTURE = 'vgg16'
VGG19_ARCHITECTURE = 'vgg19'
RESNET152_ARCHITECTURE = 'resnet152'
RESNET50_ARCHITECTURE = 'resnet50'
KERAS_RESNET50_ARCHITECTURE = 'keras_resnet50'
DENSENET169_ARCHITECTURE = 'densenet169'
DENSENET121_ARCHITECTURE = 'densenet121'
DENSENET161_ARCHITECTURE = 'densenet161'
KAGGLE_ARCHITECTURE = 'kaggle'

CLASSIFICATION_MODELS = [ALEXNET_ARCHITECTURE, KERAS_RESNET50_ARCHITECTURE]

ARCHITECTURES = [ALEXNET_ARCHITECTURE,
                 ALEXNET_REGRESSION_ARCHITECTURE,
                 VGG16_ARCHITECTURE,
                 VGG19_ARCHITECTURE,
                 RESNET152_ARCHITECTURE,
                 RESNET50_ARCHITECTURE,
                 KERAS_RESNET50_ARCHITECTURE,
                 DENSENET169_ARCHITECTURE,
                 DENSENET121_ARCHITECTURE,
                 DENSENET161_ARCHITECTURE,
                 KAGGLE_ARCHITECTURE]

# dictionary for arcitectures-image sizes
image_sizes = {ALEXNET_ARCHITECTURE: (256, 256),
               ALEXNET_REGRESSION_ARCHITECTURE: (256, 256),
               VGG16_ARCHITECTURE: (224, 224),
               VGG19_ARCHITECTURE: (224, 224),
               RESNET152_ARCHITECTURE: (256, 256),
               RESNET50_ARCHITECTURE: (256, 256),
               KERAS_RESNET50_ARCHITECTURE: (224, 224),
               DENSENET169_ARCHITECTURE: (224, 224),
               DENSENET161_ARCHITECTURE: (224, 224),
               DENSENET121_ARCHITECTURE: (224, 224),
               KAGGLE_ARCHITECTURE: (224, 224)}

crop_sizes = {ALEXNET_ARCHITECTURE: (227, 227),
              ALEXNET_REGRESSION_ARCHITECTURE: (227, 227),
              VGG16_ARCHITECTURE: None,
              VGG19_ARCHITECTURE: None,
              RESNET152_ARCHITECTURE: (224, 224),
              RESNET50_ARCHITECTURE: (224, 224),
              KERAS_RESNET50_ARCHITECTURE: None,
              DENSENET169_ARCHITECTURE: None,
              DENSENET121_ARCHITECTURE: None,
              DENSENET161_ARCHITECTURE: None,
              KAGGLE_ARCHITECTURE: None}

color_modes = {ALEXNET_ARCHITECTURE: "rgb",
               ALEXNET_REGRESSION_ARCHITECTURE: "rgb",
               VGG16_ARCHITECTURE: "rgb",
               VGG19_ARCHITECTURE: "rgb",
               RESNET152_ARCHITECTURE: "rgb",
               RESNET50_ARCHITECTURE: "rgb",
               KERAS_RESNET50_ARCHITECTURE: "rgb",
               DENSENET169_ARCHITECTURE: "rgb",
               DENSENET161_ARCHITECTURE: "rgb",
               DENSENET121_ARCHITECTURE: "rgb",
               KAGGLE_ARCHITECTURE: "rgb"}

END_TO_END_FINE_TUNING = 'end-to-end'
PHASE_BY_PHASE_FINE_TUNING = 'phase-by-phase'
FREEZE_INITIAL_LAYERS = 'freeze-initial'

FINE_TUNING_METHODS = [END_TO_END_FINE_TUNING, PHASE_BY_PHASE_FINE_TUNING, FREEZE_INITIAL_LAYERS]


# Returns formatted current time as string
def get_time_string():
    return time.strftime('%c') + ' '


def get_l1_loss(x, y):
    return abs(K.argmax(x) - K.argmax(y))


# Get the label for a file
# For yearbook this returns a year
# For streetview this returns a (longitude, latitude) pair
def label(filename):
    m = yb_r.search(filename)
    if m is not None: return int(m.group(1))
    m = sv_r.search(filename)
    assert m is not None, "Filename '%s' malformatted" % filename
    return float(m.group(2)), float(m.group(1))


# List all the yearbook files:
#   train=True, valid=False will only list training files (for training)
#   train=False, valid=True will only list validation files (for testing)
def listYearbook(train=True, valid=True, sample=False):
    r = []
    prefix = YEARBOOK_TXT_PREFIX
    if sample:
        prefix = YEARBOOK_TXT_SAMPLE_PREFIX

    if train: r += [n.strip().split('\t') for n in open(prefix + '_train.txt', 'r')]
    if valid: r += [n.strip().split('\t') for n in open(prefix + '_valid.txt', 'r')]
    return r


# List all the streetview files
def listStreetView(train=True, valid=True, sample=False):
    r = []
    prefix = STREETVIEW_TXT_PREFIX
    if sample:
        prefix = STREETVIEW_TXT_SAMPLE_PREFIX
    if train: r += [n.strip().split('\t') for n in open(prefix + '_train.txt', 'r')]
    if valid: r += [n.strip().split('\t') for n in open(prefix + '_valid.txt', 'r')]
    return r


def testListYearbook(sample=False, input_file=None):
    r = []
    if input_file is None:
        prefix = YEARBOOK_TXT_PREFIX
        if sample:
            prefix = YEARBOOK_TXT_SAMPLE_PREFIX
        input_file = prefix + '_test.txt'

    r += [n.strip().split('\t') for n in open(input_file, 'r')]
    return r


def testListStreetView(sample=False):
    r = []
    prefix = STREETVIEW_TXT_PREFIX
    if sample:
        prefix = STREETVIEW_TXT_SAMPLE_PREFIX
    r += [n.strip().split('\t') for n in open(prefix + '_test.txt', 'r')]
    return r


try:
    from mpl_toolkits.basemap import Basemap

    basemap_params = dict(projection='merc', llcrnrlat=40.390225, urcrnrlat=52.101005, llcrnrlon=-5.786422,
                          urcrnrlon=10.540445, resolution='l')
    BM = Basemap(**basemap_params)
except:
    BM = None


# Draw some coordinates for geolocation
# This function expects a 2d numpy array (N, 2) with latutudes and longitudes in them
def drawOnMap(coordinates):
    from pylab import scatter
    import matplotlib.pyplot as plt
    assert BM is not None, "Failed to load basemap. Consider running `pip install basemap`."
    BM.drawcoastlines()
    BM.drawcountries()
    # This function expects longitude, latitude as arguments
    x, y = BM(coordinates[:, 0], coordinates[:, 1])
    scatter(x, y)
    plt.savefig('geo_xy.jpg')
    plt.show()


# Map coordinates to XY positions (useful to compute distances)
# This function expects a 2d numpy array (N, 2) with latitudes and longitudes in them
def coordinateToXY(coordinates):
    import numpy as np
    assert BM is not None, "Failed to load basemap. Consider running `pip install basemap`."
    return np.vstack(BM(coordinates[:, 0], coordinates[:, 1])).T


# inverse of the above
def XYToCoordinate(xy):
    import numpy as np
    assert BM is not None, "Failed to load basemap. Consider running `pip install basemap`."
    return np.vstack(BM(xy[:, 0], xy[:, 1], inverse=True)).T


def get_data_and_labels(data, base_path):
    """
    :param data: list of tuples of images name and year label
    :param base_path: base path where the images are present
    :return: Returns list of full image path names and list of one-hot encoding of labels

    """

    images = [path.join(base_path, item[0]) for item in data]
    labels = []

    for item in data:
        # Creating a one-hot vector for the output year label
        label_vec = np.zeros(NUM_CLASSES_YEARBOOK)
        label_vec[int(item[1]) - 1900] = 1

        labels.append(label_vec)

    return images, np.array(labels)


def get_streetview_data_and_labels(data, base_path):
    """
    :param data: list of tuples of images name, lats and longs
    :param base_path: base path where the images are present
    :return: Returns list of full image path names and list of lats and longs

    """

    images = [path.join(base_path, item[0]) for item in data]
    gps = [[float(item[1]), float(item[2])] for item in data]

    return images, np.array(gps)


def get_streetview_data_and_labels_one_hot(data, base_path, width, height):
    """
    :param data: list of tuples of images name, lats and longs
    :param base_path: base path where the images are present
    :return: Returns list of full image path names and list of lats and longs

    """

    N = len(data)
    images = [path.join(base_path, item[0]) for item in data]
    labels = []

    num_classes = width * height
    min_x, max_x, min_y, max_y = get_min_max_xy_geo()

    coordinates = np.zeros((N, 2))
    for i in range(N):
        item = data[i]
        coordinates[i, 0] = item[1]
        coordinates[i, 1] = item[2]

    xy_coordinates = coordinateToXY(coordinates)

    # label_dict = {}
    # for i in range(num_classes):
    #     label_dict[i] = 0

    for item in xy_coordinates:
        # Creating a one-hot vector for the output year label
        label_vec = np.zeros(num_classes)
        int_label = get_int_label_from_xy(width, height, item[0], item[1], min_x, max_x, min_y, max_y)
        # label_dict[int_label] += 1
        label_vec[int(int_label)] = 1
        labels.append(label_vec)

    # print label_dict

    # for key, value in sorted(label_dict.iteritems(), key=lambda (k, v): (v, k)):
    #     print "%s: %s" % (key, value)

    return images, np.array(labels)


def preprocess_image_batch(image_paths, architecture, out=None):
    """
    Consistent pre-processing of images batches

    :param architecture: type of architecture (Resnet|VGG16|AlexNet)
    :param image_paths: iterable: images to process
    :param out: append output to this iterable if specified

    """
    img_list = []
    # rgb_mean = calculate_mean_of_images(image_paths, image_sizes[architecture])
    rgb_mean = [123.68, 116.779, 103.939]
    for im_path in image_paths:
        img = imread(im_path, mode='RGB')
        img_size = image_sizes[architecture]
        if img_size:
            img = imresize(img, img_size)

        img = img.astype('float32')
        # We normalize the colors (in RGB space) with the empirical means on the training set
        img[:, :, 0] -= rgb_mean[0]
        img[:, :, 1] -= rgb_mean[1]
        img[:, :, 2] -= rgb_mean[2]
        # We permute the colors to get them in the BGR order
        color_mode = color_modes[architecture]
        if color_mode == 'bgr':
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
        img = img.transpose((2, 0, 1))

        crop_size = crop_sizes[architecture]
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


def is_using_gpu():
    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = np.random.RandomState(22)
    x = shared(np.asarray(rng.rand(vlen), config.floatX))
    f = function([], tensor.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print("Result is %s" % (r,))
    if np.any([isinstance(x.op, tensor.Elemwise) and
                       ('Gpu' not in type(x.op).__name__)
               for x in f.maker.fgraph.toposort()]):
        return False
    else:
        return True


def calculate_mean_of_images(image_paths, img_size=None):
    """
    Consistent preprocessing of images batches

    :param image_paths: iterable: images to process
    :param crop_size: tuple: crop images if specified
    :param img_size: tuple: resize images if specified
    :param color_mode: Use rgb or change to bgr mode based on type of model you want to use
    :param out: append output to this iterable if specified
    """

    global_sums = np.zeros(3)
    count = 0
    for im_path in image_paths:
        count += 1
        img = imread(im_path, mode='RGB')
        if img_size:
            img = imresize(img, img_size)

        img = img.astype('float32')
        # np.avg
        local_sums = np.mean(img, axis=(0, 1))  # image is (256,256,3) hence mean across (0,1)
        global_sums += local_sums

        # if count % 100 == 0:
        #   print(str(count) + ' of ' + str(len(image_paths)) + ' complete.')

    # print 'images mean: ',
    # print(global_sums / count)
    return (global_sums / count)


# train_data = listYearbook(True, False)
# valid_data = listYearbook(False, True)
#
# train_images, train_labels = get_data_and_labels(train_data, YEARBOOK_TRAIN_PATH)
# valid_images, valid_labels = get_data_and_labels(valid_data, YEARBOOK_VALID_PATH)
#
# print_mean_of_images(image_paths=train_images, img_size=(256, 256))


def chunks(l, m, n, architecture):
    """Yield successive n-sized chunks from l and m."""
    for i in range(0, len(l), n):
        yield preprocess_image_batch(l[i:i + n], architecture), m[i: i + n]


def chunks_test(l, m, n, architecture):
    """Yield successive n-sized chunks from l and m."""
    for i in range(0, len(l), n):
        yield preprocess_image_batch(l[i:i + n], architecture), m[i: i + n]


# Evaluate L1 distance on valid data for yearbook dataset
def evaluateYearbookFromModel(model, architecture, sample=False):
    valid_data = listYearbook(False, True, sample)
    valid_images = [path.join(YEARBOOK_VALID_PATH, item[0]) for item in valid_data]
    valid_years = [int(item[1]) for item in valid_data]

    total_count = len(valid_data)
    l1_dist = 0.0
    batch_size = 128
    count = 0
    print(get_time_string() + 'Total validation data: ' + str(total_count))

    for x_chunk, y_chunk in chunks(valid_images, valid_years, batch_size, architecture):
        print(get_time_string() + 'Validating ' + str(count + 1) + ' - ' + str(count + batch_size))
        predictions = model.predict(x_chunk)
        years = np.array([np.argmax(p) + 1900 for p in predictions])
        l1_dist += np.sum(abs(years - y_chunk))
        count += batch_size

    l1_dist /= total_count
    print(get_time_string() + 'L1 distance for validation set: ' + str(l1_dist))
    return l1_dist

# Evaluate L1 distance on valid data for yearbook dataset by ensembling list of models
# Right now, it calculates mean, median and closest to mean L1 distances
def evaluateYearbookFromEnsembledModels(models_architectures_tuples, sample=False):
    valid_data = listYearbook(False, True, sample)
    valid_images = [path.join(YEARBOOK_VALID_PATH, item[0]) for item in valid_data]
    valid_years = [int(item[1]) for item in valid_data]

    total_count = len(valid_data)
    batch_size = 128
    print(get_time_string() + 'Total validation data: ' + str(total_count))

    # Matrix of predictions where each column corresponds to one architecture
    mat = np.zeros((total_count, len(models_architectures_tuples)))
    mat2 = np.zeros((total_count, NUM_CLASSES_YEARBOOK, len(models_architectures_tuples)))
    i = 0

    for (model, architecture) in models_architectures_tuples:
        count = 0
        print(get_time_string() + 'Starting validation for architecture ' + architecture)
        years_full = np.empty(0)  # Contains predictions for the entire validation set
        for x_chunk, y_chunk in chunks(valid_images, valid_years, batch_size, architecture):
            batch_len = len(x_chunk)
            print(get_time_string() + 'Validating ' + str(count + 1) + ' - ' + str(count + batch_size))
            predictions = model.predict(x_chunk)
            years = np.array([np.argmax(p) + 1900 for p in predictions])
            years_full = np.concatenate((years_full, years), axis=0)
            mat2[count: count + batch_len, :, i] = predictions
            count += batch_len
        mat[:, i] = years_full
        i += 1

    calculate_metrics_over_argmax(mat, total_count, valid_years)
    calculate_argmax_over_metrics(mat2, total_count, np.array(valid_years))


# Evaluate L1 distance on valid data for yearbook dataset by ensembling list of models
# Right now, it calculates mean, median and closest to mean L1 distances
def evaluateYearbookFromEnsembledModelsMultiple(models_map, individual_models_2d, sample=False):
    valid_data = listYearbook(False, True, sample)
    valid_images = [path.join(YEARBOOK_VALID_PATH, item[0]) for item in valid_data]
    valid_years = [int(item[1]) for item in valid_data]

    total_count = len(valid_data)
    batch_size = 128
    print(get_time_string() + 'Total validation data: ' + str(total_count))

    print(get_time_string() + 'Calculating predictions for each architecture..')
    predictions_map = {}
    for models_1d in individual_models_2d:
        for model_checkpoint in models_1d:
            if model_checkpoint not in predictions_map:  # If not already predicted for this model
                count = 0
                architecture = model_checkpoint.split(':')[0]
                model = models_map[model_checkpoint]
                print(get_time_string() + 'Starting validation for model_checkpoint ' + model_checkpoint)
                years_full = np.empty(0)  # Contains predictions for the entire validation set
                for x_chunk, y_chunk in chunks(valid_images, valid_years, batch_size, architecture):
                    batch_len = len(x_chunk)
                    print(get_time_string() + 'Validating ' + str(count + 1) + ' - ' + str(count + batch_size))
                    predictions = model.predict(x_chunk)
                    years = np.array([np.argmax(p) + 1900 for p in predictions])
                    years_full = np.concatenate((years_full, years), axis=0)
                    count += batch_len
                predictions_map[model_checkpoint] = years_full

    for models_1d in individual_models_2d:
        print(get_time_string() + 'Calculating ensembled L1 for the models: ' + str(models_1d))
        # Matrix of predictions where each column corresponds to one architecture
        mat = np.zeros((total_count, len(models_1d)))
        i = 0
        for model_checkpoint in models_1d:
            mat[:, i] = predictions_map[model_checkpoint]
            i += 1
        calculate_metrics_over_argmax(mat, total_count, valid_years)


def evaluateYearbookFromEnsembledModelsMultipleFromPredictions(predictions_map, individual_models_2d, sample=False):
    valid_data = listYearbook(False, True, sample)
    valid_years = [int(item[1]) for item in valid_data]

    total_count = len(valid_data)
    print(get_time_string() + 'Total validation data: ' + str(total_count))

    for models_1d in individual_models_2d:
        print(get_time_string() + 'Calculating ensembled L1 for the models: ' + str(models_1d))
        # Matrix of predictions where each column corresponds to one architecture
        mat = np.zeros((total_count, len(models_1d)))
        i = 0
        for model_checkpoint in models_1d:
            mat[:, i] = predictions_map[model_checkpoint]
            i += 1
        calculate_metrics_over_argmax(mat, total_count, valid_years)


def testYearbookFromEnsembledModelsMultiple(models_map, individual_models_2d, sample=False,
                                            input_file=None, output_file_suffix=None):
    test_list = testListYearbook(sample=sample, input_file=input_file)

    # Hack
    if output_file_suffix is not None and 'valid' in output_file_suffix:
        test_images = [path.join(YEARBOOK_VALID_PATH, item[0]) for item in test_list]
    else:
        test_images = [path.join(YEARBOOK_TEST_PATH, item[0]) for item in test_list]

    total_count = len(test_images)
    batch_size = 128
    print(get_time_string() + 'Total test data: ' + str(total_count))

    print(get_time_string() + 'Calculating predictions for each architecture..')
    predictions_map = {}
    for models_1d in individual_models_2d:
        for model_checkpoint in models_1d:
            if model_checkpoint not in predictions_map:  # If not already predicted for this model
                count = 0
                architecture = model_checkpoint.split(':')[0]
                model = models_map[model_checkpoint]
                print(get_time_string() + 'Starting testing for model_checkpoint ' + model_checkpoint)
                years_full = np.empty(0)  # Contains predictions for the entire validation set
                for x_chunk, _ in chunks(test_images, test_list, batch_size, architecture):
                    batch_len = len(x_chunk)
                    print(get_time_string() + 'Testing ' + str(count + 1) + ' - ' + str(count + batch_size))
                    predictions = model.predict(x_chunk)
                    years = np.array([np.argmax(p) + 1900 for p in predictions])
                    years_full = np.concatenate((years_full, years), axis=0)
                    count += batch_len
                predictions_map[model_checkpoint] = years_full

    for models_1d in individual_models_2d:
        test_file_suffix = '--'.join(models_1d)

        print(get_time_string() + 'Calculating ensembled L1 for the models: ' + str(models_1d))
        # Matrix of predictions where each column corresponds to one architecture
        mat = np.zeros((total_count, len(models_1d)))
        i = 0
        for model_checkpoint in models_1d:
            mat[:, i] = predictions_map[model_checkpoint]
            i += 1
        test_calculate_metrics_over_argmax(mat, total_count, test_list, test_file_suffix)


def testYearbookFromEnsembledModelsMultipleFromPredictions(predictions_map, individual_models_2d, sample=False,
                                                           input_file=None, output_file_suffix=None):
    test_list = testListYearbook(sample=sample, input_file=input_file)

    # Hack
    if output_file_suffix is not None and 'valid' in output_file_suffix:
        test_images = [path.join(YEARBOOK_VALID_PATH, item[0]) for item in test_list]
    else:
        test_images = [path.join(YEARBOOK_TEST_PATH, item[0]) for item in test_list]

    total_count = len(test_list)
    print(get_time_string() + 'Total test data: ' + str(total_count))

    for models_1d in individual_models_2d:
        models_wo_checkpoint = [e.split(':')[0] for e in models_1d]
        test_file_suffix = '--'.join(models_wo_checkpoint)
        if output_file_suffix is not None:
            test_file_suffix = test_file_suffix + '-' + output_file_suffix

        print(get_time_string() + 'Calculating ensembled L1 for the models: ' + str(models_1d))
        # Matrix of predictions where each column corresponds to one architecture
        mat = np.zeros((total_count, len(models_1d)))
        i = 0
        for model_checkpoint in models_1d:
            mat[:, i] = predictions_map[model_checkpoint]
            i += 1
        test_calculate_metrics_over_argmax(mat, total_count, test_list, test_file_suffix)


# Evaluate L1 distance on valid data for geolocation dataset by ensembling list of models
# Right now, it calculates mean and median L1 distances
def evaluateGeoLocationFromEnsembledModels(models_architectures_tuples, sample=False, width=10, height=10):
    valid_data = listStreetView(False, True, sample)
    valid_images = [path.join(STREETVIEW_VALID_PATH, item[0]) for item in valid_data]
    valid_gps = [[float(item[1]), float(item[2])] for item in valid_data]
    min_x, max_x, min_y, max_y = get_min_max_xy_geo()

    total_count = len(valid_data)
    batch_size = 128
    print(get_time_string() + 'Total validation data: ' + str(total_count))

    # Matrix of predictions where each column corresponds to one architecture
    mat = np.zeros((total_count, NUM_CLASSES_GEOLOCATION, len(models_architectures_tuples)))
    i = 0

    for (model, architecture) in models_architectures_tuples:
        count = 0
        print(get_time_string() + 'Starting validation for architecture ' + architecture)
        for x_chunk, y_chunk in chunks(valid_images, valid_gps, batch_size, architecture):
            batch_len = len(x_chunk)
            print(get_time_string() + 'Validating ' + str(count + 1) + ' - ' + str(count + batch_size))
            predictions = model.predict(x_chunk)
            if architecture not in CLASSIFICATION_MODELS:
                latslongs = np.array([[p[0], p[1]] for p in predictions])
            else:
                int_labels = np.array([np.argmax(p) for p in predictions])
                xy_coordinates = np.zeros((batch_len, 2))
                for j in range(batch_len):
                    int_label = int_labels[j]
                    x, y = get_xy_from_int_label(width, height, int_label, min_x, max_x, min_y, max_y)
                    xy_coordinates[j, 0] = x
                    xy_coordinates[j, 1] = y
                latslongs = XYToCoordinate(xy_coordinates)

            mat[count: count + batch_len, :, i] = latslongs
            count += batch_len
        i += 1

    calculate_ensembled_l1_geolocation(mat, total_count, valid_gps)

def testGeoLocationFromEnsembledModels(models_architectures_tuples, sample=False, width=10, height=10):
    test_data = testListStreetView(sample)
    test_images = [path.join(STREETVIEW_TEST_PATH, item[0]) for item in test_data]
    min_x, max_x, min_y, max_y = get_min_max_xy_geo()

    total_count = len(test_data)
    batch_size = 128
    print(get_time_string() + 'Total test data: ' + str(total_count))

    output_file = STREETVIEW_TEST_LABEL_PATH
    output = open(output_file, 'w')

    # Matrix of predictions where each column corresponds to one architecture
    mat = np.zeros((total_count, NUM_CLASSES_GEOLOCATION, len(models_architectures_tuples)))
    i = 0

    for (model, architecture) in models_architectures_tuples:
        count = 0
        print(get_time_string() + 'Starting validation for architecture ' + architecture)
        for x_chunk, y_chunk in chunks(test_images, test_data, batch_size, architecture):
            batch_len = len(x_chunk)
            print(get_time_string() + 'Testing ' + str(count + 1) + ' - ' + str(count + batch_size))
            predictions = model.predict(x_chunk)
            if architecture not in CLASSIFICATION_MODELS:
                latslongs = np.array([[p[0], p[1]] for p in predictions])
            else:
                int_labels = np.array([np.argmax(p) for p in predictions])
                xy_coordinates = np.zeros((batch_len, 2))
                for j in range(batch_len):
                    int_label = int_labels[j]
                    x, y = get_xy_from_int_label(width, height, int_label, min_x, max_x, min_y, max_y)
                    xy_coordinates[j, 0] = x
                    xy_coordinates[j, 1] = y
                latslongs = XYToCoordinate(xy_coordinates)

            mat[count: count + batch_len, :, i] = latslongs
            count += batch_len
        i += 1

    mean_latslongs = np.mean(mat, axis=2)
    for i in range(len(mean_latslongs)):
        out_string = test_data[i][0] + '\t' + str(mean_latslongs[i][0]) + '\t' + str(mean_latslongs[i][1]) + '\n'
        output.write(out_string)


def calculate_ensembled_l1_geolocation(mat, total_count, valid_gps):
    print(get_time_string() + 'Calculating argmax over metrics..')

    mean_latslongs = np.mean(mat, axis=2)
    median_latslongs = np.median(mat, axis=2)

    mean_l1_dist = 0.0
    median_l1_dist = 0.0

    for i in range(0, total_count):
        mean_l1_dist += dist(mean_latslongs[i][0], mean_latslongs[i][1], valid_gps[i][0], valid_gps[i][1])
        median_l1_dist += dist(median_latslongs[i][0], median_latslongs[i][1], valid_gps[i][0], valid_gps[i][1])

    mean_l1_dist /= total_count
    median_l1_dist /= total_count
    print(get_time_string() + 'L1 distance for validation set: ' + 'mean l1 distance: ' + str(
        mean_l1_dist) + 'median l1 distance' + str(median_l1_dist))


def getYearbookTestOutputFile(checkpoint_file, output_file_suffix=None):
    # checkpoint_ext = '.h5'
    # checkpoint_file_wo_ext = checkpoint_file.split(checkpoint_ext)[0]

    output_path = YEARBOOK_TEST_LABEL_PATH
    ext = '.txt'
    output_path_wo_ext = output_path.split(ext)[0]

    if output_file_suffix is None:
        return output_path_wo_ext + '-' + checkpoint_file + ext

    return output_path_wo_ext + '-' + checkpoint_file + '-' + output_file_suffix + ext


def test_calculate_metrics_over_argmax(mat, total_count, image_names, test_file_suffix):
    print(get_time_string() + 'Testing: Calculating metrics over argmax..')

    output_file_mean = getYearbookTestOutputFile(test_file_suffix + '--mean')
    output_mean = open(output_file_mean, 'w')

    output_file_median = getYearbookTestOutputFile(test_file_suffix + '--median')
    output_median = open(output_file_median, 'w')

    output_file_close_mean = getYearbookTestOutputFile(test_file_suffix + '--close_mean')
    output_close_mean = open(output_file_close_mean, 'w')

    print(get_time_string() + 'Writing output labels to file: ' + output_file_mean)
    print(get_time_string() + 'Writing output labels to file: ' + output_file_median)
    print(get_time_string() + 'Writing output labels to file: ' + output_file_close_mean)

    for i in range(total_count):
        m = mat[i, :]  # 1-d array with predictions for a particular image from different architectures
        mean = int(round(np.mean(m)))
        closest_to_mean = mean
        median = int(round(np.median(m)))

        mx = 10000
        for x in np.nditer(m):
            if abs(x - mean) < mx:
                mx = abs(x - mean)
                closest_to_mean = x

        out_string = image_names[i][0] + '\t' + str(mean) + '\n'
        output_mean.write(out_string)

        out_string = image_names[i][0] + '\t' + str(median) + '\n'
        output_median.write(out_string)

        out_string = image_names[i][0] + '\t' + str(closest_to_mean) + '\n'
        output_close_mean.write(out_string)

    output_mean.close()
    output_median.close()
    output_close_mean.close()


def calculate_metrics_over_argmax(mat, total_count, valid_years):
    print(get_time_string() + 'Calculating metrics over argmax..')

    l1_dist_mean = 0.0
    l1_dist_median = 0.0
    l1_dist_closest_to_mean = 0.0

    for i in range(total_count):
        m = mat[i, :]  # 1-d array with predictions for a particular image from different architectures
        mean = np.mean(m)
        closest_to_mean = mean
        median = np.median(m)

        mx = 10000
        for x in np.nditer(m):
            if abs(x - mean) < mx:
                mx = abs(x - mean)
                closest_to_mean = x

        l1_dist_mean += abs(int(round(mean)) - valid_years[i])
        l1_dist_median += abs(int(round(median)) - valid_years[i])
        l1_dist_closest_to_mean += abs(closest_to_mean - valid_years[i])

    l1_dist_mean /= total_count
    l1_dist_median /= total_count
    l1_dist_closest_to_mean /= total_count

    print(get_time_string() + 'L1 distance for validation set: [mean, median, closest to mean] = [' +
          str(l1_dist_mean) + ', ' + str(l1_dist_median) + ', ' + str(l1_dist_closest_to_mean) + ']')


def _get_l1_distance_argmax(s, valid_years):
    predicted_indices = np.argmax(s, axis=1)
    predicted_years = np.array([e + 1900 for e in predicted_indices])
    return np.sum(abs(predicted_years - valid_years))


def _get_l1_distance_argmin(s, valid_years):
    predicted_indices = np.argmin(s, axis=1)
    predicted_years = np.array([e + 1900 for e in predicted_indices])
    return np.sum(abs(predicted_years - valid_years))


def calculate_argmax_over_metrics(mat, total_count, valid_years):
    print(get_time_string() + 'Calculating argmax over metrics..')

    l1_sum = _get_l1_distance_argmax(np.sum(mat, axis=2), valid_years) / float(total_count)
    l1_mean = _get_l1_distance_argmax(np.mean(mat, axis=2), valid_years) / float(total_count)
    l1_median = _get_l1_distance_argmax(np.median(mat, axis=2), valid_years) / float(total_count)
    l1_max = _get_l1_distance_argmax(np.max(mat, axis=2), valid_years) / float(total_count)

    l1_std = _get_l1_distance_argmin(np.std(mat, axis=2), valid_years) / float(total_count)

    print(get_time_string() + 'L1 distance for validation set: [sum, mean, median, std, max] = [' +
          str(l1_sum) + ', ' + str(l1_mean) + ', ' + str(l1_median) + ', ' + str(l1_std) + ', ' + str(l1_max) + ']')


# Evaluate L1 distance on valid data for geolocation dataset
def evaluateStreetviewFromModel(model, architecture, sample=False):
    valid_data = listStreetView(False, True, sample)
    valid_images = [path.join(STREETVIEW_VALID_PATH, item[0]) for item in valid_data]
    valid_gps = [[float(item[1]), float(item[2])] for item in valid_data]

    total_count = len(valid_data)
    l1_dist = 0.0
    batch_size = 128
    count = 0
    print(get_time_string() + 'Total validation data: ' + str(total_count))

    for x_chunk, y_chunk in chunks(valid_images, valid_gps, batch_size, architecture):
        print(get_time_string() + 'validating ' + str(count + 1) + ' - ' + str(count + batch_size))
        predictions = model.predict(x_chunk)
        latslongs = np.array([[p[0], p[1]] for p in predictions])
        for i in range(0, len(y_chunk)):
            l1_dist += dist(latslongs[i][0], latslongs[i][1], y_chunk[i][0], y_chunk[i][1])
        count += batch_size

    l1_dist /= total_count
    print(get_time_string() + 'L1 distance for validation set: ' + str(l1_dist))
    return l1_dist


# Evaluate L1 distance on valid data for geolocation dataset
def evaluateStreetviewFromModelClassification(model, architecture, width, height, sample=False):
    min_x, max_x, min_y, max_y = get_min_max_xy_geo()

    valid_data = listStreetView(False, True, sample)
    valid_images = [path.join(STREETVIEW_VALID_PATH, item[0]) for item in valid_data]
    valid_gps = [[float(item[1]), float(item[2])] for item in valid_data]

    total_count = len(valid_data)
    l1_dist = 0.0
    batch_size = 128
    count = 0
    print(get_time_string() + 'Total validation data: ' + str(total_count))

    for x_chunk, y_chunk in chunks(valid_images, valid_gps, batch_size, architecture):
        batch_len = len(x_chunk)
        print(get_time_string() + 'validating ' + str(count + 1) + ' - ' + str(count + batch_size))
        predictions = model.predict(x_chunk)
        int_labels = np.array([np.argmax(p) for p in predictions])
        xy_coordinates = np.zeros((batch_len, 2))
        for i in range(batch_len):
            int_label = int_labels[i]
            x, y = get_xy_from_int_label(width, height, int_label, min_x, max_x, min_y, max_y)
            xy_coordinates[i, 0] = x
            xy_coordinates[i, 1] = y

        coordinates = XYToCoordinate(xy_coordinates)
        for i in range(batch_len):
            l1_dist += dist(coordinates[i][0], coordinates[i][1], y_chunk[i][0], y_chunk[i][1])
        count += batch_size

    l1_dist /= total_count
    print(get_time_string() + 'L1 distance for validation set: ' + str(l1_dist))
    return l1_dist


def numToRadians(x):
    return x / 180.0 * pi


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


def print_line():
    print('-' * 100)


def dist_between_points(x1, y1, x2, y2):
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


def get_min_max_xy_geo():
    # get train and validation data
    train_data = listStreetView(True, False, False)

    train_images, train_gps = get_streetview_data_and_labels(train_data, STREETVIEW_TRAIN_PATH)

    train_xy = coordinateToXY(train_gps)

    min_x = np.min(train_xy[:, 0], axis=0)
    max_x = np.max(train_xy[:, 0], axis=0)
    min_y = np.min(train_xy[:, 1], axis=0)
    max_y = np.max(train_xy[:, 1], axis=0)

    return min_x, max_x, min_y, max_y


def get_int_label_from_xy(width, height, x, y, min_x, max_x, min_y, max_y):
    """
    :param width: number of segments in x-coordinate
    :param height: number of segments in y-coordinate
    :param x: x-coordinate of point
    :param y: y-coordinate of point
    :param min_x: min x-coordinate in training set
    :param max_x: max x-coordinate in training set
    :param min_y: min y-coordinate in training set
    :param max_y: max y-coordinate in training set
    :return: Returns a single label which maps the point (x, y) in the grid to an integer

    For example, (min_x, min_y) will be mapped to 0 and (max_x, max_y) will be mapped to width * height - 1

    """

    x_size_per_segment = int(ceil(abs(max_x - min_x) / width))
    y_size_per_segment = int(ceil(abs(max_y - min_y) / height))

    col = int(floor(abs(x - min_x) / x_size_per_segment))
    row = int(floor(abs(y - min_y) / y_size_per_segment))

    return row * width + col


def get_xy_from_int_label(width, height, int_label, min_x, max_x, min_y, max_y):
    """
    :param width: number of segments in x-coordinate
    :param height: number of segments in y-coordinate
    :param int_label: the classification label
    :param min_x: min x-coordinate in training set
    :param max_x: max x-coordinate in training set
    :param min_y: min y-coordinate in training set
    :param max_y: max y-coordinate in training set
    :return: Returns a (x, y) coordinate belonging to the classification label

    For example, (min_x, min_y) will be mapped to 0 and (max_x, max_y) will be mapped to width * height - 1

    """

    x_size_per_segment = int(ceil(abs(max_x - min_x) / width))
    y_size_per_segment = int(ceil(abs(max_y - min_y) / height))

    row = int(floor(int_label / width))
    col = int(int_label % width)

    x = col * x_size_per_segment + x_size_per_segment / 2.0
    y = row * y_size_per_segment + y_size_per_segment / 2.0

    return x, y


def adhoc_testing_geo():
    # get train and validation data
    train_data = listStreetView(True, False, False)

    train_images, train_gps = get_streetview_data_and_labels(train_data, STREETVIEW_TRAIN_PATH)

    train_xy = coordinateToXY(train_gps)
    print(train_xy.shape)
    print(np.min(train_xy[:, 0], axis=0), np.max(train_xy[:, 0], axis=0))
    print(np.min(train_xy[:, 1], axis=0), np.max(train_xy[:, 1], axis=0))
    print(np.min(train_xy, axis=0), np.max(train_xy, axis=0))

    print(train_xy[1:5, :])
    print(train_gps[1:5, :])

    lat1 = train_gps[1, 0]
    lon1 = train_gps[1, 1]
    lat2 = train_gps[2, 0]
    lon2 = train_gps[2, 1]

    x1 = train_xy[1, 0]
    y1 = train_xy[1, 1]
    x2 = train_xy[2, 0]
    y2 = train_xy[2, 1]

    print(dist(lat1, lon1, lat2, lon2))
    print(dist_between_points(x1, y1, x2, y2))

    minx = np.min(train_xy[:, 0], axis=0)
    maxx = np.max(train_xy[:, 0], axis=0)
    miny = np.min(train_xy[:, 1], axis=0)
    maxy = np.max(train_xy[:, 1], axis=0)

    minlat = np.min(train_gps[:, 0], axis=0)
    maxlat = np.max(train_gps[:, 0], axis=0)
    minlon = np.min(train_gps[:, 1], axis=0)
    maxlon = np.max(train_gps[:, 1], axis=0)

    print(dist(minlat, minlon, maxlat, maxlon))
    print(dist_between_points(minx, miny, maxx, maxy))

    drawOnMap(train_gps)


# adhoc_testing_geo()

# get train and validation data
# train_data = listStreetView(True, False, False)
#
# train_images, train_gps = get_streetview_data_and_labels_one_hot(train_data, STREETVIEW_TRAIN_PATH, 20, 20)
# print len(train_images)
# print len(train_gps)
#
# print train_gps[0]
