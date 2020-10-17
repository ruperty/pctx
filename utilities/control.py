# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:53:14 2020

@author: ryoung
"""


import rutils.rmath as rm


def sigmoid(r, p, g, scale):
    e=r-p
    return rm.sigmoid( e, g, scale),e

def proportional(r, p, g):
    e=r-p
    return e*g,e



def integrator(r, p, g, s, o):
    e=r-p
    o +=  (g * e - o) / s;
    return o,e

