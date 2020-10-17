# -*- coding: utf-8 -*-
"""
Created on Sat May 16 16:11:40 2020

@author: rupert
"""


from tensorflow.python.keras.metrics import MeanMetricWrapper

from pct.dl.functions import squared_sum_error
from pct.dl.functions import my_mean_squared_error
from pct.dl.functions import root_squared_sum_error


class MyMeanSquaredError(MeanMetricWrapper):

  def __init__(self, name='my_mse', dtype=None):
    super(MyMeanSquaredError, self).__init__(
        my_mean_squared_error, name, dtype=dtype)


class SquaredSumError(MeanMetricWrapper):

  def __init__(self, name='sse', dtype=None):
    super(SquaredSumError, self).__init__(
        squared_sum_error, name, dtype=dtype)

class RootSquaredSumError(MeanMetricWrapper):

  def __init__(self, name='rsse', dtype=None):
    super(RootSquaredSumError, self).__init__(
        root_squared_sum_error, name, dtype=dtype)


# Aliases.


rsse = RSSE = root_squared_sum_error
sse = SSE = squared_sum_error
my_mse = MY_MSE = my_mean_squared_error