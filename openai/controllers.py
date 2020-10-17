# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:53:14 2020

@author: ryoung
"""


#from plotly.offline import plot
import plotly.graph_objs as go
import numpy as np
import pct.openai.utils as ou
import pct.utilities.rmath as rm
from pct.utilities.errors import root_sum_squared_error
from pct.openai.cpplots import add_cartpolepoints_to_widget
from pct.openai.cpplots import add_cartpole_positions_to_widget
import math


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



def moving_controller(ctr, pole_position_ref, state, gains, slow, figures, position_figure, serror, sfactor, scale, prev_power, plot, cart_position_ref):
    action =1
    cart_position,    cart_velocity ,    pole_angle , pole_velocity =ou.get_obs(state)        
    print(f'{pole_angle:.3f} {pole_velocity:.3f} {cart_position:.3f} {cart_velocity:.3f}')    
    """
    test=0.2
    cart_position=test
    cart_velocity=test 
    pole_angle=test
    pole_velocity=test        
    """
    pole_position_gain, pole_angle_gain,pole_velocity_gain,cart_position_gain,cart_velocity_gain = ou.get_gains(gains)

    pole_position=cart_position+math.sin(pole_angle)
    pole_angle_ref,pole_position_error=sigmoid(pole_position_ref, pole_position, pole_position_gain, scale)
    
    pole_velocity_ref,pole_angle_error=proportional(pole_angle_ref, pole_angle, pole_angle_gain)
    
    cart_position_ref,pole_velocity_error=integrator(pole_velocity_ref, pole_velocity, pole_velocity_gain, slow, cart_position)
    #cart_position_ref,pole_velocity_error=integrator(pole_velocity_ref, pole_velocity, pole_velocity_gain, slow, cart_position_ref)
    
    cart_velocity_ref,cart_position_error=proportional(cart_position_ref, cart_position, cart_position_gain)
    power,cart_velocity_error=proportional(cart_velocity_ref, cart_velocity, cart_velocity_gain)
    power= rm.smooth(power, prev_power, 0.75)
    print(f'pole_position {pole_position_ref:.3f} {pole_position:.3f} {pole_position_error:.3f} {pole_angle_ref:.3f}')    
    print(f'pole_angle {pole_angle_ref:.3f} {pole_angle:.3f} {pole_angle_error:.3f} {pole_velocity_ref:.3f}')    
    print(f'pole_velocity {pole_velocity_ref:.3f} {pole_velocity:.3f} {pole_velocity_error:.3f} {cart_position_ref:.3f}  **')    
    print(f'cart_position {cart_position_ref:.3f} {cart_position:.3f} {cart_position_error:.3f} {cart_velocity_ref:.3f}')    
    print(f'cart_velocity {cart_velocity_ref:.3f} {cart_velocity:.3f} {cart_velocity_error:.3f} {power:.3f}')    

    if power>=0:
        action=0

    print(action)        

        
    error=root_sum_squared_error([pole_angle_error,pole_velocity_error,cart_position_error,cart_velocity_error])
    serror=rm.smooth( error, serror, sfactor) 
    if plot:
        if type(figures)==go.FigureWidget:
            add_cartpolepoints_to_widget(figures, 2, ctr, pole_angle_ref, pole_angle, pole_velocity_ref, pole_velocity, 
                       cart_position_ref, cart_position, cart_velocity_ref, cart_velocity, action, error, serror)        
        else:    
            figures.add_points( ctr, pole_angle_ref, pole_angle, pole_velocity_ref, pole_velocity, 
                       cart_position_ref, cart_position, cart_velocity_ref, cart_velocity, action, error, serror)
        
        if type(position_figure)==go.FigureWidget:        
            add_cartpole_positions_to_widget(position_figure, ctr, pole_position_ref, pole_position)
        else:    
            position_figure.add_points( ctr, pole_position_ref, pole_position)

    return action, serror, pole_position, power, cart_position_ref

def moving_pole_angle_controller(ctr, pole_angle_ref, state, gains, slow, figures, serror, sfactor, plot):
    action =1
    cart_position,    cart_velocity ,    pole_angle , pole_velocity =ou.get_obs(state)
    pole_angle_gain,pole_velocity_gain,cart_position_gain,cart_velocity_gain = ou.get_gains(gains)
    
    pole_velocity_ref,pole_angle_error=proportional(pole_angle_ref, pole_angle, pole_angle_gain)
    cart_position_ref,pole_velocity_error=integrator(pole_velocity_ref, pole_velocity, pole_velocity_gain, slow, cart_position)
    cart_velocity_ref,cart_position_error=proportional(cart_position_ref, cart_position, cart_position_gain)
    power,cart_velocity_error=proportional(cart_velocity_ref, cart_velocity, cart_velocity_gain)
    if power>=0:
        action=0
        
    error=root_sum_squared_error([pole_angle_error,pole_velocity_error,cart_position_error,cart_velocity_error])
    serror=rm.smooth( error, serror, sfactor) 
    
    if plot:
        figures.add_points( ctr, pole_angle_ref, pole_angle, pole_velocity_ref, pole_velocity, 
                   cart_position_ref, cart_position, cart_velocity_ref, cart_velocity, action, error, serror)
    

    return action, serror



def controller(ctr, pole_angle_ref, state, gains, figures, serror, sum_error_ma, sfactor, prev_power):
    action =1
    cart_position,    cart_velocity ,    pole_angle , pole_velocity =ou.get_obs(state)
    pole_angle_gain,pole_velocity_gain,cart_position_gain,cart_velocity_gain = ou.get_gains(gains)
    
    pole_velocity_ref,pole_angle_error=proportional(pole_angle_ref, pole_angle, pole_angle_gain)
    cart_position_ref,pole_velocity_error=proportional(pole_velocity_ref, pole_velocity, pole_velocity_gain)
    cart_velocity_ref,cart_position_error=proportional(cart_position_ref, cart_position, cart_position_gain)
    power,cart_velocity_error=proportional(cart_velocity_ref, cart_velocity, cart_velocity_gain)
    print(power, end=" ")
    #power = prev_power + 0.05 * power
    power= rm.smooth(power, prev_power, 0.8)
    print(power)
    if power>=0:
        action=0
        
    sum_error = pole_angle_error+pole_velocity_error+cart_position_error+cart_velocity_error
    sum_error_ma=rm.smooth( sum_error, sum_error_ma, sfactor) 
    
    error=root_sum_squared_error([pole_angle_error,pole_velocity_error,cart_position_error,cart_velocity_error])
    serror=rm.smooth( error, serror, sfactor) 
    
    figures.add_points( ctr, pole_angle_ref, pole_angle, pole_velocity_ref, pole_velocity, 
                   cart_position_ref, cart_position, cart_velocity_ref, cart_velocity, action, sum_error_ma, serror)
       

    return action, serror, sum_error_ma, power

def update_angle_figure(ctr, figure, pole_angle_ref, pole_angle):
    newx = np.array(figure.data[0].x)
    newx=np.append(   newx,ctr) 
    figure.data[0].x=newx

    newy0 = np.array(figure.data[0].y)
    newy0 =np.append(   newy0,pole_angle_ref)
    figure.data[0].y=newy0
    #print(newy0)

    figure.data[1].x=newx
    
    newy1 = np.array(figure.data[1].y)
    newy1 =np.append(   newy1,pole_angle) 
    figure.data[1].y=newy1


"""
def plot_figures(figures):
    for figure in figures:
        plot(figure)
    
    

def set_figures():
    #plots
    angle_fig = go.Figure()
    pole_angle_ref_scatt = angle_fig.add_scatter(x=[],y=[])
    pole_angle_scatt = angle_fig.add_scatter(x=[],y=[])
    #angle_fig
    
    return [angle_fig]
"""