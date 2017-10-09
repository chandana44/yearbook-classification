from os import path
import util
import numpy as np
import argparse
from skimage.io import imread
from util import *
import csv


def load(image_path):
    # TODO:load image and process if you want to do any
    img = imread(image_path)
    return img


class Predictor:
    def __init__(self, dataset, model_arch_tuples):
        self.model_architecture_tuples = model_arch_tuples
        self.dataset = dataset

    # baseline 1 which calculates the median of the train data and return each time
    def yearbook_baseline(self):
        # Load all training data
        train_list = listYearbook(train=True, valid=False)

        # Get all the labels
        years = np.array([float(y[1]) for y in train_list])
        med = np.median(years, axis=0)
        return [med]

    # Compute the median.
    # We do this in the projective space of the map instead of longitude/latitude,
    # as France is almost flat and euclidean distances in the projective space are
    # close enough to spherical distances.
    def streetview_baseline(self):
        # Load all training data
        train_list = listStreetView(train=True, valid=False)

        # Get all the labels
        coord = np.array([(float(y[1]), float(y[2])) for y in train_list])
        xy = coordinateToXY(coord)
        med = np.median(xy, axis=0, keepdims=True)
        med_coord = np.squeeze(XYToCoordinate(med))
        return med_coord

    def predict(self, image_path):
        if self.dataset == 'yearbook':
            mat = np.zeros(len(self.model_architecture_tuples))
            i=0

            for (model, architecture) in self.model_architecture_tuples:
                processed_image = preprocess_image_batch(np.array([image_path]),architecture)
                prediction = model.predict(processed_image)
                year = np.array([np.argmax(prediction) + 1900])
                mat[i] = year
                i += 1

            result = np.mean(mat)
            #result = np.median(mat)
        elif self.dataset == 'geolocation':
            min_x, max_x, min_y, max_y = get_min_max_xy_geo()

            # Matrix of predictions where each column corresponds to one architecture
            mat = np.zeros((NUM_CLASSES_GEOLOCATION, len(self.model_architecture_tuples)))
            i = 0

            for (model, architecture) in self.model_architecture_tuples:
                processed_image = preprocess_image_batch(np.array([image_path]),architecture)
                prediction = model.predict(processed_image)
                if architecture not in CLASSIFICATION_MODELS:
                    latslongs = prediction
                else:
                    int_label = np.array([np.argmax(prediction)])
                    xy_coordinates = np.zeros(2)
                    x, y = get_xy_from_int_label(10, 10, int_label, min_x, max_x, min_y, max_y)
                    xy_coordinates[0] = x
                    xy_coordinates[1] = y
                    latslongs = XYToCoordinate(xy_coordinates)
                mat[ :, i] = latslongs
                i += 1

            mean_latslongs = np.mean(mat, axis=1)
            result = mean_latslongs
        else:
            raise Exception('Unknown dataset type')

        return result
