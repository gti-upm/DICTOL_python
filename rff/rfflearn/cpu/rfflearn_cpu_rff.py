#!/usr/bin/env python3
#
# Python module for Random Fourier Features with random matrix for CPU.
##################################################### SOURCE START #####################################################

import numpy as np
import sklearn.svm
import sklearn.multiclass

from .rfflearn_cpu_common import Base

### Random Fourier Features.
class RFF(Base):

    ### Constractor. Save hyper parameters as member variables.
    def __init__(self, rand_mat_type = "rff", dim_kernel = 128, std_kernel = 0.1, W = None):
        super().__init__(rand_mat_type, dim_kernel, std_kernel, W)
        self.rand_mat_type = rand_mat_type

    ### Run training, that is, extract feature vectors.
    def fit(self, X):
        self.set_weight(X.shape[1])
        return self

    def rff_compute(self, X):
        if self.rand_mat_type == "rff":
            feat = self.conv(X)  # Feauture dimension is double than dim_kernel
        elif self.rand_mat_type == "rp":
            feat = self.convRP(X)
        return feat


##################################################### SOURCE FINISH ####################################################
# Author:
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
