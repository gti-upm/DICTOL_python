from dictol import SRC, utils
import numpy as np
import time

print('\n================================================================')
print('Sparse Representation-based Classification (SRC)')

# Select dataset and prepare train and test subsets
# Available datasets (see dictol/data/):
# - myYaleB
# - myARgender
# - myARreduce
# - myFlower
print('\nLoad dataset ================================================================')
dataset = 'myYaleB'
N_train = 15  # Training images per class to build dictionary.
Y_train, Y_test, label_train, label_test = utils.train_test_split(dataset, N_train)
print(f'Num training images: {Y_train.shape[1]}')
print(f'Num test images: {Y_test.shape[1]}')
print(f'Class ids: {set(label_train)}')

# Compute dictionary as the straightforward concatenation of features vector from images.
print('\n Dictionary building ================================================================')
src = SRC.SRC(lamb=0.01)  # Sparsity regularizer in the underlying lasso minimization problem.
src.fit(Y_train, label_train)
print(f'Num Classes: {src.num_classes}')
print(f'Atoms in dictionary: {src.D.shape[1]}')

# Image Classification using SRC on test data
print('\n Classification ================================================================')
start = time.time()
pred = src.predict(Y_test, iterations=100)
end = time.time()
print(f'Elapse time for prediction(s): {end - start}')
acc = np.sum(pred == label_test) / float(len(label_test))
print('Accuracy = {:.2f} %'.format(100 * acc))

