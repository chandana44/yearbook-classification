# Feel free to modify this to your needs. We will not rely on your util.py
from os import path

# If you want this to work do not move this file
SRC_PATH = path.dirname(path.abspath(__file__))
DATA_PATH = path.join(SRC_PATH, '..', 'data')

YEARBOOK_PATH = path.join(DATA_PATH, "yearbook", "yearbook")
STREETVIEW_PATH = path.join(DATA_PATH, "geo", "geo")

import re
yb_r = re.compile("(\d\d\d\d)_(.*)_(.*)_(.*)_(.*)")
sv_r = re.compile("([+-]?\d*\.\d*)_([+-]?\d*\.\d*)_\d*_-004")

# Get the label for a file
# For yearbook this returns a year
# For streetview this returns a (longitude, latitude) pair
def label(filename):
  m = yb_r.search(filename)
  if m is not None: return int(m.group(1))
  m = sv_r.search(filename)
  assert m is not None, "Filename '%s' malformatted"%filename
  return float(m.group(2)), float(m.group(1))

# List all the yearbook files:
#   train=True, valid=False will only list training files (for training)
#   train=False, valid=True will only list validation files (for testing)
def listYearbook(train=True, valid=True):
  r = []
  if train: r = r + [n.strip().split('\t') for n in open(YEARBOOK_PATH+'_train.txt','r')]
  if valid: r = r + [n.strip().split('\t') for n in open(YEARBOOK_PATH+'_valid.txt','r')]
  return r

# List all the streetview files
def listStreetView(train=True, valid=True):
  r = []
  if train: r = r + [n.strip().split('\t') for n in open(STREETVIEW_PATH+'_train.txt','r')]
  if valid: r = r + [n.strip().split('\t') for n in open(STREETVIEW_PATH+'_valid.txt','r')]
  return r

def testListYearbook():
  r = []
  r = r + [n.strip().split('\t') for n in open(YEARBOOK_PATH+'_test.txt','r')]
  return r

def testListStreetView():
  r = []
  r = r + [n.strip().split('\t') for n in open(STREETVIEW_PATH+'_test.txt','r')]
  return r

try:
  from mpl_toolkits.basemap import Basemap
  basemap_params = dict(projection='merc',llcrnrlat=40.390225,urcrnrlat=52.101005, llcrnrlon=-5.786422,urcrnrlon=10.540445, resolution='l')
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
  x,y = BM(coordinates[:,0], coordinates[:,1])
  scatter( x, y )
  plt.show()

# Map coordinates to XY positions (useful to compute distances)
# This function expects a 2d numpy array (N, 2) with latitudes and longitudes in them
def coordinateToXY(coordinates):
  import numpy as np
  assert BM is not None, "Failed to load basemap. Consider running `pip install basemap`."
  return np.vstack( BM(coordinates[:,0], coordinates[:,1]) ).T

# inverse of the above
def XYToCoordinate(xy):
  import numpy as np
  assert BM is not None, "Failed to load basemap. Consider running `pip install basemap`."
  return np.vstack( BM(xy[:,0], xy[:,1], inverse=True) ).T

