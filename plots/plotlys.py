# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:37:54 2020

@author: ryoung
"""

from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly


colors = plotly.colors.DEFAULT_PLOTLY_COLORS



class BasePlots():
    #def add_point(self, subplot, trace, x, y):    
    #    self.figure.add_point(subplot, trace, x, y)
 
    def show(self):
        plot(self.figure)
 
    def getFigure(self):
        return self.figure
    
    def add_data(self):
        index = 0
        for datalist in self.figure.data:
            datalist.x = self.data[index][0]
            datalist.y = self.data[index][1]
            index=index+1


class SingleScatterPlot(BasePlots):
    def __init__(self, title, trace_names, traces=2, width=None, height=None):
        self.title=title
        ctr=0
        color_ctr = 0
        self.figure=go.Figure()
        self.data = list()

        for trace in range(traces):
            xy = list()
            xy.append(list())
            xy.append(list())
            self.data.append(xy)
            self.figure.add_trace(go.Scatter(x=[], y=[], name=trace_names[ctr], 
                                line=dict(width=1, color=colors[color_ctr]), 
                                marker=dict(size=3,color=colors[color_ctr]),
                                mode='markers+lines'))
            color_ctr=color_ctr+1
            ctr=ctr+1

        self.figure.update_layout(height=height, width=width, title_text=title)
        
    def add_point_to_figure(self, index, x, y):            
        newx = self.figure.data[index].x + (x,)
        self.figure.data[index].x=newx

        newy = self.figure.data[index].y + (y,)
        self.figure.data[index].y=newy
    
    
    def add_point(self, index, x, y):            
        self.data[index][0].append(x)
        self.data[index][1].append(y)
        
    def get_data_trace(self, index):
        return  self.data[index]
    


class MultipleScatterSubPlots(BasePlots):
    def __init__(self, title, subplots_titles, trace_names, rows=2, cols=1, traces=2, width=None, height=None):
        self.title=title
        self.figure = make_subplots(rows, cols, subplot_titles=(subplots_titles))
        self.traces_per_subplot=traces
        self.data = list()#[] #np.empty(rows*cols*traces)
 
        ctr=0
        for row in range(rows):
            for col in range(cols):
                color_ctr = 0
                for trace in range(traces):
                    xy = list()
                    xy.append(list())
                    xy.append(list())
                    self.data.append(xy)
                    self.figure.add_trace(go.Scatter(x=[], y=[], name=trace_names[ctr], 
                                        line=dict(width=1, color=colors[color_ctr]), 
                                        marker=dict(size=3,color=colors[color_ctr]),
                                        mode='markers+lines',legendgroup=row), 
                                        row=row+1, col=col+1)
                    color_ctr=color_ctr+1
                    ctr=ctr+1

        self.figure.update_layout(height=height, width=width, title_text=title)
       
    
    def add_point(self, subplot, trace, x, y):    
        index = subplot*self.traces_per_subplot+trace
        
        self.data[index][0].append(x)
        self.data[index][1].append(y)
        
        
    def add_point_to_figure(self, subplot, trace, x, y):    
        index = subplot*self.traces_per_subplot+trace
        
        newx = self.figure.data[index].x + (x,)
        self.figure.data[index].x=newx

        newy = self.figure.data[index].y + (y,)
        self.figure.data[index].y=newy
        
  
    
 
    
def add_point_to_subplot_widget(widget, traces_per_subplot, subplot, trace, x, y):
    index = subplot*traces_per_subplot+trace
    newx = widget.data[index].x + (x,)
    widget.data[index].x=newx

    newy = widget.data[index].y + (y,)
    widget.data[index].y=newy




    
def add_point_to_widget(widget, index, x, y):
    newx = widget.data[index].x + (x,)
    widget.data[index].x=newx

    newy = widget.data[index].y + (y,)
    widget.data[index].y=newy




















