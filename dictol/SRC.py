from __future__ import print_function
from . import utils
from .optimize import Lasso
from . import base
import numpy as np


class SRC(base.BaseModel):
    def __init__(self, lamb=0.01):
        self.lamb = lamb
        self.D = None
        self.train_range = None
        self.C = None
        self.num_classes = None

    def fit(self, Y_train, label_train):
        self.D = Y_train
        self.train_range = utils.label_to_range(label_train)
        self.num_classes = len(self.train_range) - 1

    def predict(self, Y, iterations=100, mean_spars=False):
        lasso = Lasso(self.D, self.lamb)
        lasso.fit(Y, iterations=iterations)  # X = arg min_X 0.5*||Y - DX||_F^2 + lambd||X||_1
        X = lasso.coef_
        if mean_spars:
            # Mean sparsity using a the accumulated energy
            threshold = 0.9  # 90% of the energy
            sorted_desc = -np.sort(-np.abs(X), axis=0)
            cs = np.cumsum(sorted_desc, axis=0)
            cs_bin = cs/cs[-1,:] > threshold
            sparse_level = np.zeros((1,cs_bin.shape[1]))
            for i in range(cs_bin.shape[1]):
                sparse_level[0,i] = np.where(cs_bin[:,i])[0][0]
            mean_sparsity = np.mean(sparse_level)
            std_sparsity = np.std(sparse_level)
            # Mean sparsity using a threshold
            '''
            mean_sparsity = np.mean(np.sum(np.abs(X) >= 1 / (self.D.shape[1]/self.num_classes), axis=0))
            std_sparsity = np.std(np.sum(np.abs(X) >= 1 / (self.D.shape[1]/self.num_classes), axis=0))
            '''
            print(f'Mean sparsity: {mean_sparsity} out of {self.D.shape[1]}')
            print(f'Std sparsity: {std_sparsity}')
        E = np.zeros((self.num_classes, Y.shape[1]))
        for i in range(self.num_classes):
            Xi = utils.get_block_row(X, i, self.train_range)  # Extract coefficients corresponding to class i
            Di = utils.get_block_col(self.D, i, self.train_range) # Extract atoms corresponding to class i
            R = Y - np.dot(Di, Xi)
            E[i, :] = (R*R).sum(axis=0)  # Residual function per class
        return utils.vec(np.argmin(E, axis=0) + 1)  # The estimated class is that associated with the less residual error


def mini_test_unit():
    print('\n================================================================')
    print('Mini Unit test: Sparse Representation-based Classification (SRC)')
    dataset = 'myYaleB'
    N_train = 2
    Y_train, Y_test, label_train, label_test, *other = utils.train_test_split(dataset, N_train)
    src = SRC(lamb=0.01)
    src.fit(Y_train, label_train)
    src.evaluate(Y_test, label_test)


def test_unit():
    print('\n================================================================')
    print('Unit test: Sparse Representation-based Classification (SRC)')
    dataset = 'myYaleB'
    N_train = 15
    Y_train, Y_test, label_train, label_test, *other = utils.train_test_split(dataset, N_train)
    src = SRC(lamb=0.01)
    src.fit(Y_train, label_train)
    src.evaluate(Y_test, label_test)


if __name__ == '__main__':
    mini_test_unit()
