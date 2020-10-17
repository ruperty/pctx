# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:42:36 2020

@author: rupert
"""

from numpy import exp

def smooth( newVal, oldVal, weight) :
    return newVal * (1 - weight) + oldVal * weight;



def sigmoid(x, range, scale) :
    return -range / 2 + range / (1 + exp(-x * scale / range));
    


class Counter(object):

  def __init__(self, limit=1000, init=0, step=1, plot=10, print=100):
      self.limit=limit
      self.counter=init
      self.step=step
      self.plot=plot
      self.print=print

  def __call__(self):
      self.counter+=self.step
      return self.counter
  
  def get(self):
      return self.counter
    
  def get_limit(self):
      return self.limit

  def set_limit(self, limit):
      self.limit=limit

  

"""
rang=0.4
scale=.5

for x in range(-10,10):
    print(x/2.5, sigmoid(x/2.5, rang, scale))
"""