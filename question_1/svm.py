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
from generate_data import X_test, X_train, y_test, y_train, x1_lim, x2_lim
import k_fold as kf


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

def plot_binary_classification_results(ax, predictions, labels):
    # Get indices of the four decision scenarios:
    # True Negatives
    tn = np.argwhere((predictions == -1) & (y_test == -1))
    # False Positives
    fp = np.argwhere((predictions == 1) & (y_test == -1))
    # False Negative Probability
    fn = np.argwhere((predictions == -1) & (y_test == 1))
    # True Positive Probability
    tp = np.argwhere((predictions == 1) & (y_test == 1))

    # class -1 circle, class 1 +, correct green, incorrect red
    ax.plot(X_test[tn, 0], X_test[tn, 1], 'og', label="Correct Class -1")
    ax.plot(X_test[fp, 0], X_test[fp, 1], 'or', label="Incorrect Class -1")
    ax.plot(X_test[fn, 0], X_test[fn, 1], '+r', label="Incorrect Class 1")
    ax.plot(X_test[tp, 0], X_test[tp, 1], '+g', label="Correct Class 1")

classifier = SVC(C=kf.C_best, kernel='rbf', gamma=kf.gamma_best)

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

incorrect_ind = np.argwhere(y_test != predictions)
prob_error_test = len(incorrect_ind) / N_test
print("SVM Probability error on the test data set: %.4f\n" % prob_error_test)

fig, ax = plt.subplots(figsize=(10, 10))
plot_binary_classification_results(ax, predictions, y_test)
# Define region of interest by data limits
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
x_span = np.linspace(x_min, x_max, num=200)
y_span = np.linspace(y_min, y_max, num=200)
xx, yy = np.meshgrid(x_span, y_span)

grid = np.c_[xx.ravel(), yy.ravel()]

# Z matrix are the SVM classifier predictions
Z = classifier.predict(grid).reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.viridis, alpha=0.25)

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("SVM Decisions (RED incorrect) on Test Set")
plt.legend()
plt.tight_layout()
plt.show()

# Simply using sklearn confusion matrix
print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(predictions, y_test)
conf_display = ConfusionMatrixDisplay.from_predictions(predictions, y_test, display_labels=['-1', '+1'], colorbar=False)
plt.ylabel("Predicted Labels")
plt.xlabel("True Labels")
plt.show()