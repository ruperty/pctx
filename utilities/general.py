# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 22:16:41 2020

@author: rupert
"""



class Counter(object):

  def __init__(self, limit):
      self.limit=limit

  def __call__(self):
      return self.limit
  
  def set_limit(self, limit):
      self.limit=limit
