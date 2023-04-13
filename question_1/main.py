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
from svm import grid, xx, yy, plot_binary_classification_results


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

from mlp import TwoLayerMLP, model_predict, model_train

P_list = [2, 4, 8, 16, 24, 32, 48, 64, 128]

# Converting -1/+1 labels into a binary format, suitable for the MLP loss function
lb = LabelBinarizer()
y_train_binary = lb.fit_transform(y_train)[:, 0]

P_best = kf.k_fold_cv_perceptrons(kf.K, P_list, X_train, y_train_binary)

# Number of times to re-train same model with random re-initializations
num_restarts = 10

# Convert numpy structures to PyTorch tensors, necessary data types
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train_binary)
            
# List of trained MlPs for later testing
restart_mlps = []
restart_losses = []
# Remove chances of falling into suboptimal local minima
for r in range(num_restarts):
    model = TwoLayerMLP(X_train.shape[1], P_best)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # Trained model
    model, loss = model_train(model, X_train_tensor, y_train_tensor, optimizer)
    restart_mlps.append(model)
    restart_losses.append(loss.detach().item())

# Choose best model from multiple restarts to list
best_mlp = restart_mlps[np.argmin(restart_losses)]

X_test_tensor = torch.FloatTensor(X_test)

# Evaluate the neural network on the test set
prediction_probs = model_predict(best_mlp, X_test_tensor)
# Decision boundary set to 0.5, hence rounding up sigmoid outputs
predictions = np.round(prediction_probs)
# Return back to original encoding
predictions = lb.inverse_transform(predictions)

# Get indices of correct and incorrect labels
incorrect_ind = np.argwhere(y_test != predictions)
prob_error_test = len(incorrect_ind) / N_test
print("MLP Pr(error) on the test data set: %.4f\n" % prob_error_test)

fig, ax = plt.subplots(figsize=(10, 10));

plot_binary_classification_results(ax, predictions, y_test)

grid_tensor = torch.FloatTensor(grid)
# Make predictions across region of interest from before when plotting the SVM decision surfaces
best_mlp.eval()
Z = best_mlp(grid_tensor).detach().numpy()
Z = lb.inverse_transform(np.round(Z)).reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.viridis, alpha=0.25)

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("MLP Decisions (RED incorrect) on Test Set")
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