from os import path
import util
import numpy as np
import argparse
from skimage.io import imread
from util import *
import csv

def load(image_path):
	#TODO:load image and process if you want to do any
	img=imread(image_path)
	return img
	
class Predictor:
	DATASET_TYPE = 'yearbook'
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

		img = load(image_path)

		#TODO: load model

		#TODO: predict model and return result either in geolocation format or yearbook format
		# depending on the dataset you are using
		if self.DATASET_TYPE == 'geolocation':
			result = self.streetview_baseline() #for geolocation
		elif self.DATASET_TYPE == 'yearbook':
			result = self.yearbook_baseline() #for yearbook
		return result
		
	


