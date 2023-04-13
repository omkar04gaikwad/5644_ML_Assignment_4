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
from variables import n, mix_pdf, N_test, N_train


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



def generate_multiring_dataset(N, n, pdf_params):
    # Output samples and labels
    X = np.zeros([N, n])
    # Note that the labels are either -1 or +1, binary classification
    labels = np.ones(N)
    
    # Decide randomly which samples will come from each class
    indices = np.random.rand(N) < pdf_params['prior']
    # Reassign random samples to the negative class values (to -1)
    labels[indices] = -1
    num_neg = sum(indices)

    # Create mixture distribution
    theta = np.random.uniform(low=-np.pi, high=np.pi, size=N)
    uniform_component = np.array([np.cos(theta), np.sin(theta)]).T

    # Positive class samples
    X[~indices] = pdf_params['r+'] * uniform_component[~indices] + mvn.rvs(pdf_params['mu'], pdf_params['Sigma'],
                                                                           N-num_neg)
    # Negative class samples
    X[indices] = pdf_params['r-'] * uniform_component[indices] + mvn.rvs(pdf_params['mu'], pdf_params['Sigma'],
                                                                         num_neg)
    return X, labels


X_train, y_train = generate_multiring_dataset(N_train, n, mix_pdf)
X_test, y_test = generate_multiring_dataset(N_test, n, mix_pdf)


fig, ax = plt.subplots(2, 1, figsize=(10,10))

ax[0].set_title("Training Set")
ax[0].plot(X_train[y_train==-1, 0], X_train[y_train==-1, 1], 'bo', label="Class -1")
ax[0].plot(X_train[y_train==1, 0], X_train[y_train==1, 1], 'k+', label="Class 1")
ax[0].set_xlabel(r"$x_1$")
ax[0].set_ylabel(r"$x_2$")
ax[0].legend()

ax[1].set_title("Test Set")
ax[1].plot(X_test[y_test==-1, 0], X_test[y_test==-1, 1], 'bo', label="Class -1")
ax[1].plot(X_test[y_test==1, 0], X_test[y_test==1, 1], 'k+', label="Class 1")
ax[1].set_xlabel(r"$x_1$")
ax[1].set_ylabel(r"$x_2$")
ax[1].legend()

# Using test set samples to limit axes
x1_lim = (floor(np.min(X_test[:,0])), ceil(np.max(X_test[:,0])))
x2_lim = (floor(np.min(X_test[:,1])), ceil(np.max(X_test[:,1])))
# Keep axis-equal so there is new skewed perspective due to a greater range along one axis
plt.setp(ax, xlim=x1_lim, ylim=x2_lim)
plt.tight_layout()
plt.show()