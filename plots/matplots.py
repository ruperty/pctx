# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 20:27:36 2020

@author: rupert
"""

import matplotlib
import matplotlib.pyplot as plt


class RegressionPlotter(object):

  def __init__(self):
    self.xs, self.ls, self.Ws, self.bs, self.dWs, self.dbs = [],[],[],[],[], []
    
  def add_data(self, epoch, model, optimizer):
    self.Ws.append(model.W.numpy())
    self.bs.append(model.b.numpy())
    self.xs.append(epoch)
    self.ls.append(optimizer.previous_loss)    
    self.dWs.append(optimizer.dWeights[0])
    self.dbs.append(optimizer.dWeights[1])
    if epoch > 100:
        self.xs.pop(0)
        self.ls.pop(0)
        self.dWs.pop(0)
        self.dbs.pop(0)



class Plotter(object):

  def __init__(self, width, height):
    self.xs, self.ls, self.Ws, self.bs, self.dWs, self.dbs = [],[],[],[],[], []
    self.fig = plt.figure(figsize=[width, height])
    plt.title ='Custom Training'
    self.ax1 = plt.subplot(121)
    self.ax2 = plt.subplot(222)
    self.ax3 = plt.subplot(224)
    
  def add_data(self, epoch, model, optimizer):
    self.Ws.append(model.W.numpy())
    self.bs.append(model.b.numpy())
    self.xs.append(epoch)
    self.ls.append(optimizer.previous_loss)    
    self.dWs.append(optimizer.dWeights[0])
    self.dbs.append(optimizer.dWeights[1])
    if epoch > 100:
        self.xs.pop(0)
        self.ls.pop(0)
        self.dWs.pop(0)
        self.dbs.pop(0)
    #print(len(self.xs))
    #print(len(self.ls))

        
  def draw(self, model):
    self.ax1.clear()
    self.ax1.set_xlim([-5,5])
    self.ax1.set_ylim([-25,25])
    self.ax1.scatter(model.inputs, model.outputs, c='b')
    self.ax1.scatter(model.inputs, model(model.inputs), c='r')
    self.ax1.set_title('Data')
    self.ax1.set_xlabel('x')
    self.ax1.set_ylabel('y')
    self.ax1.margins(x=5,y=10)
    
    
    xupper=self.xs[-1]
    if self.xs[-1]==0:
        xupper=1
    self.ax2.clear()
    self.ax2.set_ylim([0,50])
    self.ax2.set_xlim([self.xs[0],xupper])
    self.ax2.plot(self.xs, self.ls, 'r')
    self.ax2.set_title('Loss')
    self.ax2.set_xlabel('iteration')
    self.ax2.set_ylabel('loss')
    self.ax2.margins(5,5)
    
    self.ax3.clear()
    self.ax3.set_ylim([-1,1])
    #xlower, x = self.xs[-1]
    #if x-100<0:
     #   xlower=0
    self.ax3.set_xlim([self.xs[0],xupper])
    self.ax3.plot(self.xs, self.dWs, 'r')
    self.ax3.plot(self.xs, self.dbs, 'b')
    self.ax3.set_title('dWeights')
    self.ax3.set_xlabel('iteration')
    self.ax3.set_ylabel('dw')
    self.ax3.margins(5,5)
    
    plt.tight_layout()
        
        
 

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    #print(backend)
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)
        
        
class SubPlotter(object):

  def __init__(self, width, height, title, plotsconfig=[["title", "xlabel", "ylabel", 1, 0, 111]]):
      

    self.fig = plt.figure(figsize=[width, height])
    self.fig.canvas.set_window_title(title)
    
    self.plots=[]
    self.colors=['b',  'r', 'g', 'c', 'm', 'y', 'k', 'w']

    for plotconfig in plotsconfig:
        #print(plotconfig)
        ys=[]
        for line in range(plotconfig[3]):
            ys.append([])
        plot = dict([("title", plotconfig[0]), ("xlabel", plotconfig[1]), 
                     ("ylabel", plotconfig[2]), ("window", plotconfig[4]), 
                     ("subplot", plt.subplot(plotconfig[5])), ("x", []), ("ys", ys)])
        #print(plot)

        self.plots.append(plot)
    
  def add_data(self, index, x, ys):
    #print(ys)
    plot= self.plots[index]
    plot["x"].append(x)
    for i in range(len(ys)):
        plot["ys"][i].append(ys[i])
    window = plot["window"]
    if window > 0:
        if x > window:
            plot["x"].pop(0)
            for i in range(len(ys)):
                plot["ys"][i].pop(0)
            
        
  def show(self):
    plt.show()
        
  def draw(self):
    for plot in self.plots:
        plot["subplot"].clear()
        #print(plot["ys"])
        ctr=0
        for y in plot["ys"]:
            plot["subplot"].plot(plot["x"], y, self.colors[ctr])
            ctr+=1
        plot["subplot"].set_title(plot["title"])
        plot["subplot"].set_xlabel(plot["xlabel"])
        plot["subplot"].set_ylabel(plot["ylabel"])
        #self.ax1.margins(x=5,y=10)

    plt.tight_layout()


class CartpolePlotter(object):

  def __init__(self, width, height, title, xlabel, ylabel):
    self.global_error, self.xs= [],[]
    self.fig = plt.figure(figsize=[width, height])
    self.title=title
    self.xlabel=xlabel
    self.ylabel=ylabel
    self.ax1 = plt.subplot(121)
    
  def add_data(self, epoch, error):
    self.xs.append(epoch)
    self.global_error.append(error)    

        
  def draw(self):
    self.ax1.clear()
    self.ax1.plot(self.xs, self.global_error)
    self.ax1.set_title(self.title)
    self.ax1.set_xlabel(self.xlabel)
    self.ax1.set_ylabel(self.ylabel)
    #self.ax1.margins(x=5,y=10)
    plt.tight_layout()