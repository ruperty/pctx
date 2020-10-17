# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:45:04 2020

@author: rupert
"""


#import tensorflow as tf 

from tensorflow.python.keras.utils import losses_utils

from tensorflow.python.keras.losses import LossFunctionWrapper

from pct.dl.functions import squared_sum_error
from pct.dl.functions import my_mean_squared_error
from pct.dl.functions import root_squared_sum_error
from pct.dl.functions import root_sum_squared_error



class MyMeanSquaredError(LossFunctionWrapper):

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='my_mse'):
    super(MyMeanSquaredError, self).__init__(
        my_mean_squared_error, name=name, reduction=reduction)


class SquaredSumError(LossFunctionWrapper):

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='sse'):
    super(SquaredSumError, self).__init__(
        squared_sum_error, name=name, reduction=reduction)


class RootSquaredSumError(LossFunctionWrapper):

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='rsse'):
    super(RootSquaredSumError, self).__init__(
        root_squared_sum_error, name=name, reduction=reduction)



class RootSumSquaredError(LossFunctionWrapper):

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='rsuse'):
    super(RootSumSquaredError, self).__init__(
        root_sum_squared_error, name=name, reduction=reduction)





# Aliases.

rsuse = RSUSE = root_sum_squared_error
rsse = RSSE = root_squared_sum_error
sse = SSE = squared_sum_error
my_mse = MY_MSE = my_mean_squared_error