import matplotlib.pyplot as plt # For general plotting
from matplotlib.ticker import MaxNLocator

from math import ceil, floor 

import numpy as np

from scipy.stats import multivariate_normal as mvn

from skimage.io import imread

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC

import torch
import torch.nn as nn
import torch.nn.functional as F



np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)

plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=22)   # fontsize of the figure title


def generate_feature_vector(image):
    image_np = np.array(image)
    img_indices = np.indices((image_np.shape[0], image_np.shape[1]))

    if image_np.ndim == 2:
        features = np.array([img_indices[0].flatten(), img_indices[1].flatten(), image_np.flatten()])
        min_f = np.min(features, axis=1)
        max_f = np.max(features, axis=1)
        ranges = max_f - min_f
        normalized_data = np.diag(1/ranges).dot(features - min_f[:, np.newaxis])
    elif image_np.ndim == 3:
        features = np.array([img_indices[0].flatten(), img_indices[1].flatten(),
                             image_np[..., 0].flatten(), image_np[..., 1].flatten(), image_np[...,2].flatten()])
        min_f = np.min(features, axis=1)
        max_f = np.max(features, axis=1)
        ranges = max_f - min_f
        normalized_data = np.diag(1/ranges).dot(features - min_f[:, np.newaxis])
    else:
        print("Incorrect image dimensions for feature vector")
    
    return image_np, normalized_data.T
