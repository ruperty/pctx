# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:44:52 2020

@author: ryoung
"""


import numpy as np
import plotly.graph_objects as go
from pct.plots.plotlys import  SingleScatterPlot
from pct.plots.plotlys import  MultipleScatterSubPlots
from pct.plots.plotlys import add_point_to_widget
from pct.plots.plotlys import add_point_to_subplot_widget


class BaseSpecificPlots():
    #def add_point(self, subplot, trace, x, y):    
    #    self.figure.add_point(subplot, trace, x, y)
 

    def show(self):
        self.figure.add_data()
        self.figure.show()

    def getFigure(self):
        return self.figure.getFigure()




class CartpolePositionPlot(BaseSpecificPlots):
    def __init__(self, width=None, height=None):
        trace_names=["pole_position_ref", "pole_position"]
        self.figure = SingleScatterPlot("Pole Position", trace_names,  2,  width=width, height=height)

    def add_points(self, ctr, pole_position_ref, pole_position):
        self.figure.add_point(0, ctr, pole_position_ref)
        self.figure.add_point(1, ctr, pole_position)


class CartpoleErrorsPlot(BaseSpecificPlots):
    def __init__(self, width=None, height=None):
        trace_names=["pole_angle", "pole_velocity",  "cart_position", "cart_velocity"]
        self.figure = SingleScatterPlot("Errors", trace_names,  4,  width=width, height=height)

    def add_points(self, ctr, errors):
        for i in range(len(errors)):
            self.figure.add_point(i, ctr, errors[i])


class CartpoleDataPlot(BaseSpecificPlots):
    def __init__(self, type, width=None, height=None):
        trace_names=["pole_angle", "pole_velocity",  "cart_position", "cart_velocity", "global_error"]
        self.figure = SingleScatterPlot(type, trace_names,  5,  width=width, height=height)

    def add_points(self, ctr, errors):
        for i in range(len(errors)):
            self.figure.add_point(i, ctr, errors[i])
            
    def get_data_trace(self, index):
        #print(index)
        d = self.figure.get_data_trace(index)
        #print(d)
        return d

class CartpoleControlPlots(BaseSpecificPlots):
    def __init__(self,sub_title, width=None, height=None):
        subplots_titles=["pole_angle", "pole_velocity", "cart_position", "cart_velocity", "action", "global error"]
        trace_names=["pole_angle_ref", "pole_angle", "pole_velocity_ref", "pole_velocity", 
                     "cart_position_ref","cart_position", "cart_velocity_ref", "cart_velocity",
                     "action", "", "sum", "rms"]
        self.figure = MultipleScatterSubPlots("Control Unit Values"+sub_title, subplots_titles, trace_names, 
                                              6, 1, 2,  width=width, height=height)
        #print(self.figure.data)
        
    def add_points_to_figure(self, ctr, pole_angle_ref, pole_angle, pole_velocity_ref, pole_velocity, 
                   cart_position_ref, cart_position, cart_velocity_ref, cart_velocity, action, error, serror):
        self.figure.add_point_to_figure(0, 0, ctr, np.rad2deg(pole_angle_ref))
        self.figure.add_point_to_figure(0, 1, ctr, np.rad2deg(pole_angle))
        self.figure.add_point_to_figure(1, 0, ctr, pole_velocity_ref)
        self.figure.add_point_to_figure(1, 1, ctr, pole_velocity)
        self.figure.add_point_to_figure(2, 0, ctr, cart_position_ref)
        self.figure.add_point_to_figure(2, 1, ctr, cart_position)
        self.figure.add_point_to_figure(3, 0, ctr, cart_velocity_ref)
        self.figure.add_point_to_figure(3, 1, ctr, cart_velocity)
        if action==0: action = -1
        self.figure.add_point_to_figure(4, 0, ctr, action)
        self.figure.add_point_to_figure(5, 0, ctr, error)
        self.figure.add_point_to_figure(5, 1, ctr, serror)


    def add_points(self, ctr, pole_angle_ref, pole_angle, pole_velocity_ref, pole_velocity, 
                   cart_position_ref, cart_position, cart_velocity_ref, cart_velocity, action, error, serror):
        self.figure.add_point(0, 0, ctr, np.rad2deg(pole_angle_ref))
        self.figure.add_point(0, 1, ctr, np.rad2deg(pole_angle))
        self.figure.add_point(1, 0, ctr, pole_velocity_ref)
        self.figure.add_point(1, 1, ctr, pole_velocity)
        self.figure.add_point(2, 0, ctr, cart_position_ref)
        self.figure.add_point(2, 1, ctr, cart_position)
        self.figure.add_point(3, 0, ctr, cart_velocity_ref)
        self.figure.add_point(3, 1, ctr, cart_velocity)
        if action==0: action = -1
        self.figure.add_point(4, 0, ctr, action)
        self.figure.add_point(5, 0, ctr, error)
        self.figure.add_point(5, 1, ctr, serror)
        



def add_cartpolepoints_to_widget(widget, tps, ctr, pole_angle_ref, pole_angle, pole_velocity_ref, pole_velocity, 
                   cart_position_ref, cart_position, cart_velocity_ref, cart_velocity, action, error, serror):
        add_point_to_subplot_widget(widget, tps,0, 0, ctr, np.rad2deg(pole_angle_ref))
        add_point_to_subplot_widget(widget, tps,0, 1, ctr, np.rad2deg(pole_angle))
        add_point_to_subplot_widget(widget, tps,1, 0, ctr, pole_velocity_ref)
        add_point_to_subplot_widget(widget, tps,1, 1, ctr, pole_velocity)
        add_point_to_subplot_widget(widget, tps,2, 0, ctr, cart_position_ref)
        add_point_to_subplot_widget(widget, tps,2, 1, ctr, cart_position)
        add_point_to_subplot_widget(widget, tps,3, 0, ctr, cart_velocity_ref)
        add_point_to_subplot_widget(widget, tps,3, 1, ctr, cart_velocity)
        if action==0: action = -1
        add_point_to_subplot_widget(widget, tps,4, 0, ctr, action)
        add_point_to_subplot_widget(widget, tps,5, 0, ctr, error)
        add_point_to_subplot_widget(widget, tps,5, 1, ctr, serror)
        
        
def add_cartpole_positions_to_widget(widget, ctr, pole_position_ref, pole_position):
    add_point_to_widget(widget, 0, ctr, pole_position_ref)
    add_point_to_widget(widget, 1, ctr, pole_position)

    

class CartpoleControlUnitPlots(BaseSpecificPlots):
    def __init__(self,sub_title):
        subplots_titles=["pole_angle", "pole_velocity", "cart_position", "cart_velocity"]
        trace_names=["pole_angle_ref", "pole_angle", "pole_velocity_ref", "pole_velocity", 
                     "cart_position_ref","cart_position", "cart_velocity_ref", "cart_velocity"]
        self.figure = MultipleScatterSubPlots("Control Unit Values"+sub_title, subplots_titles, trace_names, 4, 1, 2)

        
    def add_points(self, ctr, pole_angle_ref, pole_angle, pole_velocity_ref, pole_velocity, 
                   cart_position_ref, cart_position, cart_velocity_ref, cart_velocity):
        self.figure.add_point(0, 0, ctr, np.rad2deg(pole_angle_ref))
        self.figure.add_point(0, 1, ctr, np.rad2deg(pole_angle))
        self.figure.add_point(1, 0, ctr, pole_velocity_ref)
        self.figure.add_point(1, 1, ctr, pole_velocity)
        self.figure.add_point(2, 0, ctr, cart_position_ref)
        self.figure.add_point(2, 1, ctr, cart_position)
        self.figure.add_point(3, 0, ctr, cart_velocity_ref)
        self.figure.add_point(3, 1, ctr, cart_velocity)
    
        
        
class CartpoleControlValuesPlots(BaseSpecificPlots):
    def __init__(self,sub_title):
        subplots_titles=["Action", "Global Error"]
        trace_names=["action", "error"]
        self.figure = MultipleScatterSubPlots("Control Values"+sub_title, subplots_titles, trace_names, 2, 1, 1)


    def add_points(self, ctr, action, error):
        if action==0: action = -1
        self.figure.add_point(0, 0, ctr, action)
        self.figure.add_point(1, 0, ctr, error)
        
        
        

def load_3d_landscape(filename, max):
    data = np.load(filename, allow_pickle=True)
    
    type=data[0]
    other_wt_name=data[1]
    other_wt=data[2]    
    xlabel=type[0: 2]
    ylabel=type[2: 4]
    
    x_data=data[3]
    y_data=data[4]
    z_data=np.clip(data[5], 0, max)
        
    title = 'Loss Landscape {:s} - {:s} {:4.1f}'.format(type,other_wt_name,other_wt)
    scene = dict(xaxis_title=xlabel,yaxis_title=ylabel,zaxis_title='loss')
    
    fig = go.Figure(data=[go.Surface(z=z_data,x=x_data, y=y_data)])
    
    fig.update_layout(title=title, autosize=True,
                      width=800, height=800, 
                      scene = scene,
                      margin=dict(l=100, r=50, b=65, t=90))
    fig.update_yaxes(automargin=True)
    
    fig.show()
    
def scatter_plot(data, title, xtitle, ytitle):
    fig = go.Figure()
    fig.add_trace(go.Scatter(mode='lines',x=data[0], y=data[1], name=title))
    fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title=ytitle)
    fig.show()

    