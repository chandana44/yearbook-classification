# Feel free to modify this to your needs. We will not rely on your util.py
import re
import time
from os import path
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from theano import function, config, shared, tensor
from keras import backend as K
from math import sin, cos, atan2, sqrt, pi

# If you want this to work do not move this file
SRC_PATH = path.dirname(path.abspath(__file__))
DATA_PATH = path.join(SRC_PATH, '..', 'data')

YEARBOOK_PATH = path.join(DATA_PATH, "yearbook")
YEARBOOK_TXT_PREFIX = path.join(YEARBOOK_PATH, "yearbook")
YEARBOOK_TXT_SAMPLE_PREFIX = path.join(YEARBOOK_PATH, "yearbook_sample")

YEARBOOK_TRAIN_PATH = path.join(YEARBOOK_PATH, 'train')
YEARBOOK_VALID_PATH = path.join(YEARBOOK_PATH, 'valid')

STREETVIEW_PATH = path.join(DATA_PATH, "geo")
STREETVIEW_TXT_PREFIX = path.join(STREETVIEW_PATH, "geo")

STREETVIEW_TRAIN_PATH = path.join(STREETVIEW_PATH, 'train')
STREETVIEW_VALID_PATH = path.join(STREETVIEW_PATH, 'valid')

NUM_CLASSES = 118

yb_r = re.compile("(\d\d\d\d)_(.*)_(.*)_(.*)_(.*)")
sv_r = re.compile("([+-]?\d*\.\d*)_([+-]?\d*\.\d*)_\d*_-004")

# architectures
ALEXNET_ARCHITECTURE = 'alexnet'
VGG16_ARCHITECTURE = 'vgg16'
VGG19_ARCHITECTURE = 'vgg19'
RESNET152_ARCHITECTURE = 'resnet152'
RESNET50_ARCHITECTURE = 'resnet50'
DENSENET169_ARCHITECTURE = 'densenet169'
KAGGLE_ARCHITECTURE = 'kaggle'

ARCHITECTURES = [ALEXNET_ARCHITECTURE,
                 VGG16_ARCHITECTURE,
                 VGG19_ARCHITECTURE,
                 RESNET152_ARCHITECTURE,
                 RESNET50_ARCHITECTURE,
                 DENSENET169_ARCHITECTURE,
                 KAGGLE_ARCHITECTURE]



# dictionary for arcitectures-image sizes
image_sizes = {ALEXNET_ARCHITECTURE: (256, 256),
               VGG16_ARCHITECTURE: (224, 224),
               RESNET152_ARCHITECTURE: (256, 256),
               RESNET50_ARCHITECTURE: (256, 256),
               DENSENET169_ARCHITECTURE: (224, 224),
               KAGGLE_ARCHITECTURE: (224, 224)}

crop_sizes = {ALEXNET_ARCHITECTURE: (227, 227),
              VGG16_ARCHITECTURE: None,
              RESNET152_ARCHITECTURE: (224, 224),
              RESNET50_ARCHITECTURE: (224, 224),
              DENSENET169_ARCHITECTURE: None,
              KAGGLE_ARCHITECTURE: None}

color_modes = {ALEXNET_ARCHITECTURE: "rgb",
               VGG16_ARCHITECTURE: "rgb",
               RESNET152_ARCHITECTURE: "rgb",
               RESNET50_ARCHITECTURE: "rgb",
               DENSENET169_ARCHITECTURE: "rgb",
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
def listStreetView(train=True, valid=True):
    r = []
    if train: r += [n.strip().split('\t') for n in open(STREETVIEW_TXT_PREFIX + '_train.txt', 'r')]
    if valid: r += [n.strip().split('\t') for n in open(STREETVIEW_TXT_PREFIX + '_valid.txt', 'r')]
    return r


def testListYearbook(sample=False):
    r = []
    prefix = YEARBOOK_TXT_PREFIX
    if sample:
        prefix = YEARBOOK_TXT_SAMPLE_PREFIX

    r += [n.strip().split('\t') for n in open(prefix + '_test.txt', 'r')]
    return r


def testListStreetView():
    r = []
    r += [n.strip().split('\t') for n in open(STREETVIEW_TXT_PREFIX + '_test.txt', 'r')]
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
        label_vec = np.zeros(NUM_CLASSES)
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
    gps = [[item[1], item[2]] for item in data]

    return images, np.array(gps)


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
        print get_time_string() + 'validating ' + str(count+1) + ' - ' + str(count + batch_size)
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
    count = 0
    print(get_time_string() + 'Total validation data: ' + str(total_count))

    # Matrix of predictions where each column corresponds to one architecture
    mat = np.zeros((total_count, len(models_architectures_tuples)))
    i = 0

    for (model, architecture) in models_architectures_tuples:
        print(get_time_string() + 'Starting validation for architecture ' + architecture)
        years_full = np.empty(0) # Contains predictions for the entire validation set
        for x_chunk, y_chunk in chunks(valid_images, valid_years, batch_size, architecture):
            print(get_time_string() + 'Validating ' + str(count+1) + ' - ' + str(count + batch_size))
            predictions = model.predict(x_chunk)
            years = np.array([np.argmax(p) + 1900 for p in predictions])
            years_full = np.concatenate((years_full, years), axis=0)
        mat[:, i] = years_full
        i += 1

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

        l1_dist_mean += abs(mean - valid_years[i])
        l1_dist_median += abs(median - valid_years[i])
        l1_dist_closest_to_mean += abs(closest_to_mean - valid_years[i])

    l1_dist_mean /= total_count
    l1_dist_median /= total_count
    l1_dist_closest_to_mean /= total_count

    print(get_time_string() + 'L1 distance for validation set: [mean, median, closest to mean] = [' +
          str(l1_dist_mean) + ', ' + str(l1_dist_median) + ', ' + str(l1_dist_closest_to_mean) + ']')

# Evaluate L1 distance on valid data for geolocation dataset
def evaluateStreetviewFromModel(model, architecture):

    valid_data = listStreetView(False, True)
    valid_images = [path.join(STREETVIEW_VALID_PATH, item[0]) for item in valid_data]
    valid_gps = [[item[1], item[2]] for item in valid_data]

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
            l1_dist += dist(latslongs[0], latslongs[1], float(y_chunk[0]), float(y_chunk[1]))
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
