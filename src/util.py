# Feel free to modify this to your needs. We will not rely on your util.py
import numpy as np
from os import path
import re
from scipy.misc import imread
from scipy.misc import imresize

# If you want this to work do not move this file
SRC_PATH = path.dirname(path.abspath(__file__))
DATA_PATH = path.join(SRC_PATH, '..', 'data')
YEARBOOK_PATH = path.join(DATA_PATH, 'yearbook', '')
YEARBOOK_TRAIN_PATH = path.join(YEARBOOK_PATH, 'train')
YEARBOOK_VALID_PATH = path.join(YEARBOOK_PATH, 'valid')
YEARBOOK_TEST_PATH = path.join(YEARBOOK_PATH, 'test')
YEARBOOK_TEST_LABEL_PATH = path.join(SRC_PATH, '..', 'output', 'yearbook_test_label.txt')
STREETVIEW_PATH = path.join(DATA_PATH, 'geo', '')
STREETVIEW_VALID_PATH = path.join(STREETVIEW_PATH, 'train')
STREETVIEW_VALID_PATH = path.join(STREETVIEW_PATH, 'valid')
STREETVIEW_TEST_PATH = path.join(STREETVIEW_PATH, 'test')
STREETVIEW_TEST_LABEL_PATH = path.join(SRC_PATH, '..', 'output', 'geo_test_label.txt')

yb_r = re.compile("(\d\d\d\d)_(.*)_(.*)_(.*)_(.*)")
sv_r = re.compile("([+-]?\d*\.\d*)_([+-]?\d*\.\d*)_\d*_-004")


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
def listYearbook(train=True, valid=True):
    r = []
    if train: r = r + [n.strip().split('\t') for n in open(YEARBOOK_PATH + 'yearbook_train.txt', 'r')]
    if valid: r = r + [n.strip().split('\t') for n in open(YEARBOOK_PATH + 'yearbook_valid.txt', 'r')]
    return r


# List all the streetview files
def listStreetView(train=True, valid=True):
    r = []
    if train: r = r + [n.strip().split('\t') for n in open(STREETVIEW_PATH + 'geo_train.txt', 'r')]
    if valid: r = r + [n.strip().split('\t') for n in open(STREETVIEW_PATH + 'geo_valid.txt', 'r')]
    return r


def testListYearbook():
    r = []
    r = r + [n.strip().split('\t') for n in open(YEARBOOK_PATH + 'yearbook_test.txt', 'r')]
    return r


def testListStreetView():
    r = []
    r = r + [n.strip().split('\t') for n in open(STREETVIEW_PATH + 'geo_test.txt', 'r')]
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
