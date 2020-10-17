# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:54:13 2020

@author: rupert
"""


def get_gains(gains):
    return (i for i in gains)

def get_obs(observation):
    cart_position = observation[0]
    cart_velocity = observation[1]
    pole_angle = observation[2]
    pole_velocity = observation[3]
    return cart_position,    cart_velocity ,    pole_angle , pole_velocity 

class EnvState(object):

  def __init__(self):
    self.state=True
    self.action=0

  def __call__(self):
    return self.state

  def set_action(self, action):
    self.action=action

  def get_action(self):
    return self.action

  def set_state(self, state):
    self.state=state