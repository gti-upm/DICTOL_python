# TODO: no esta terminado, solo copiado de run_DLSI.py

from dictol import DLSI, utils
import numpy as np
import time

print('\n================================================================')
print('Dictionary learning with structured incoherence (DLSI)')

# Select dataset and prepare train and test subsets
# Available datasets (see dictol/data/):
# - myYaleB
# - myARgender
# - myARreduce
# - myFlower
print('\n Load dataset ================================================================')
dataset = 'myYaleB'
N_train = 15  # Training images per class to build dictionary.
Y_train, Y_test, label_train, label_test = utils.train_test_split(dataset, N_train)
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