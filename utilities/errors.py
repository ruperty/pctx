# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:42:36 2020

@author: rupert
"""

import numpy as np
import math

def root_squared_sum_error( errors ):
    return  math.sqrt(pow(np.sum( errors ),2 ) )


def root_sum_squared_error( errors ):
    return  math.sqrt(np.sum( np.square( errors ) ))
