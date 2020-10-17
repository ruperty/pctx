# -*- coding: utf-8 -*-
"""
Created on Sat May 16 16:21:52 2020

@author: rupert
"""


import tensorflow as tf

def my_mean_squared_error( Y , y_pred ):
    return tf.reduce_mean( tf.square( y_pred - Y ) )


def squared_sum_error( Y , y_pred ):
    #size = 400 #Y.shape[1]
    result =   tf.square(tf.reduce_sum ( y_pred - Y ) ) / Y[0] #10000
    return result

def root_squared_sum_error( Y , y_pred ):
    return  tf.sqrt(tf.square(tf.reduce_sum ( y_pred - Y ) ) )


def root_sum_squared_error(  Y , y_pred ):
    return  tf.sqrt(tf.reduce_sum( tf.square(  y_pred - Y ) ))



