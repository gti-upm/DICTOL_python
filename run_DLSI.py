from dictol import DLSI, utils
import numpy as np
import time
from skimage.io import imread
from skimage import io
from matplotlib import pyplot as plt
import os

print('\n================================================================')
print('Dictionary learning with structured incoherence (DLSI)')

# Select dataset and prepare train and test subsets
# Available datasets with precomputed feature vectors (see dictol/data/):
# - myYaleB
# - myARgender
# - myARreduce
# - myFlower
# Available datasets with images (see dictol/data/):
# - CroppedYale
print('\n Load dataset ================================================================')
dataset = 'CroppedYale'  #'myYaleB'
N_train = 32  # Training images per class to build dictionary. Total training samples: N_train x Num classes
dim_feat = 504  # Dimension of the image feature vectors
start = time.time()
Y_train, Y_test, label_train, label_test, le, rff = utils.train_test_split(dataset, N_train, dim_feat=dim_feat)
end = time.time()
print(f'Elapse time for loading dataset and feature computation (s): {end - start}')
print(f'Num training images: {Y_train.shape[1]}')
print(f'Num test images: {Y_test.shape[1]}')
print(f'Class ids: {set(label_train)}')


# Learn a dictionary from the training features vector obtained from images.
# lamda: Sparsity regularizer in the underlying lasso minimization problem.
# k: number of atoms per class in the dictionary
print('\n Dictionary learning ================================================================')
dlsi = DLSI.DLSI(k=10, lambd=0.001, eta=0.001)
start = time.time()
dlsi.fit(Y_train, label_train, iterations=100, verbose=True)
end = time.time()
print(f'Elapse time for dictionary learning(s): {end - start}')
print(f'Num Classes: {dlsi.num_classes}')
print(f'Atoms in dictionary: {dlsi.D.shape[1]}')


# Image Classification using SRC on test data
print('\n Classification ================================================================')
start = time.time()
pred = dlsi.predict(Y_test, iterations=100)
end = time.time()
print(f'Elapse time for prediction(s): {end - start}')
acc = np.sum(pred == label_test) / float(len(label_test))
print('Accuracy = {:.2f} %'.format(100 * acc))

# Class names from prediction
if le:
  class_id = le.inverse_transform(np.asarray(label_test)-1)
  class_pred = le.inverse_transform(pred-1)
  print(f'Actual ids for the first five test samples: {class_id[0:5]}')
  print(f'Id predictions for the first five test samples: {class_pred[0:5]}')

# Prediction over a specific image
im_path = 'dictol/data/CroppedYale/yaleB04/yaleB04_P00A+000E+45.pgm'
if rff:
    print('\n Test from one image ================================================================')
    # Computes feature vector (RFF) from image
    im = imread(im_path)                          # Read image
    io.imshow(im)
    plt.show()
    im = (im/255)-0.5
    im_vec = im.reshape((1, -1))                  # To vector
    rffeat = np.asarray(rff.rff_compute(im_vec)).transpose()  # RRF computation (on CPU)

    # Estimate label from folder name
    full_path = os.path.dirname(im_path)
    label = os.path.normpath(full_path).split(os.path.sep)[-1]
    print(f'Actual id: {label}')
    pred_one = dlsi.predict(rffeat, iterations=100)
    print(f'Predicted id: {le.inverse_transform(pred_one-1)}')
