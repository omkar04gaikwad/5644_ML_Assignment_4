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
from skimage import transform
import torch
import torch.nn as nn
import torch.nn.functional as F
from utility import generate_feature_vector
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

# Load the image
airplane_image = imread('https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/images/plain/normal/color/37073.jpg')
airplane_image = transform.resize(airplane_image, (airplane_image.shape[0] // 2, airplane_image.shape[1] // 2))
fig = plt.figure(figsize=(10,10))
plt.imshow(airplane_image)
plt.title("Airplane Image")

## Generate feature vector on the image
img_np, feature_vector = generate_feature_vector(airplane_image)   
    
# Perform EM to estimate the parameters of the GMM using fit() and default parameters
gmm = GaussianMixture(n_components=4, max_iter=400, tol=1e-3)

# Hard clustering using argmax to compute most probable component labels
gmm_predictions = gmm.fit_predict(feature_vector) 

# Assigned segment labels reshaped into an image to color-code pixels
labels_img = gmm_predictions.reshape(img_np.shape[0], img_np.shape[1])
fig = plt.figure(figsize=(10, 10))
plt.imshow(labels_img)
plt.title(r"GMM Image Segmentation Result with $K = 4$")

K_folds = 10
n_components_list = [2, 4, 6, 8, 10, 15, 20]

def k_fold_gmm_components(K, n_components_list, data):
    kf = KFold(n_splits=K, shuffle=True)
    log_lld_valid_mk = np.zeros((len(n_components_list), K))

    m = 0
    for comp in n_components_list:
        k = 0
        for train_indices, valid_indices in kf.split(data):
            gmm = GaussianMixture(n_components=comp, max_iter=400, tol=1e-3).fit(feature_vector)
            log_lld_valid_mk[m, k] = gmm.score(feature_vector)
            k += 1
        m += 1

    log_lld_valid_m = np.mean(log_lld_valid_mk, axis=1)
    best_three_ind = np.argpartition(log_lld_valid_m, -3)[-3:]
    best_three = best_three_ind[np.argsort((-log_lld_valid_m)[best_three_ind])]
    print("Best No. Cluster Components: %d" % n_components_list[best_three[0]])
    print("Log-likelihood ratio: %.3f" % np.max(log_lld_valid_m))

    fig = plt.figure(figsize=(10, 10))
    plt.plot(n_components_list, log_lld_valid_m)
    plt.title("No. components vs Cross-Validation Log-Likelihood")
    plt.xlabel(r"$K$")
    plt.ylabel("Log-Likelihood")
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()
    return [n_components_list[i] for i in best_three]
best_three_components = k_fold_gmm_components(K_folds, n_components_list, feature_vector)

# Create figure to plot all GMM segmentation results for the example image
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0,0].imshow(airplane_image)
ax[0,0].set_title("Boat Color")
ax[0,0].set_axis_off()

# Plot axis index for each clustered image 
j = 1
for comp in best_three_components:
    gmm_predictions = GaussianMixture(n_components=comp, max_iter=400, tol=1e-3).fit_predict(feature_vector)
    labels_img = gmm_predictions.reshape(img_np.shape[0], img_np.shape[1])
    
    ax[floor(j/2),j%2].imshow(labels_img)
    ax[floor(j/2),j%2].set_title(fr"Top {j} with $K = {comp}$")
    ax[floor(j/2),j%2].set_axis_off()
    j += 1
    
plt.tight_layout()
plt.show()