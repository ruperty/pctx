# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:34:42 2020

@author: rupert
"""

from PIL import Image
import numpy as np
import pct.dl.losses as pctloss
import pct.dl.metrics as pctmetrics
import pct.dl.optimizers as pctopts

import tensorflow as tf

def ecoli_continuous_train_loop(epoch, model, optimizer, counter, case, plotter, slope, lossdata, ecolifigwidge, prnt=10):
    if epoch <= counter.get_limit():
        case()
        plotter.add_data(epoch, model, optimizer)
        
        #plotter.draw(model)
        if epoch % prnt == 0 or optimizer.previous_loss <  1.05:
            print('Epoch %3d: %s %s ' %
                (epoch, model.status_string(), optimizer.status_string()))
            
        slope.append(optimizer.dlsmooth)
        lossdata.append(optimizer.previous_loss)
        ecolifigwidge.data[0].x=model.inputs
        ecolifigwidge.data[0].y=model.outputs

        ecolifigwidge.data[1].x=model.inputs
        ecolifigwidge.data[1].y=model(model.inputs)
        
        if epoch % 20 == 0 :
            ecolifigwidge.data[2].x=plotter.xs
            ecolifigwidge.data[2].y=plotter.ls

            ecolifigwidge.data[3].x=plotter.xs
            ecolifigwidge.data[3].y=plotter.dWs

            ecolifigwidge.data[4].x=plotter.xs
            ecolifigwidge.data[4].y=plotter.dbs
        
        if optimizer.previous_loss <  1.05:
            counter.set_limit(epoch) 

def ecoli_train_loop(epoch, model, optimizer, counter, case, plotter, slope, lossdata, ecolifigwidge, prnt=10):
    if epoch <= counter.get_limit():
        case()
        plotter.add_data(epoch, model, optimizer)
        
        #plotter.draw(model)
        if epoch % prnt == 0 or optimizer.previous_loss <  1.05:
            print('Epoch %3d: %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %10.3f' %
                (epoch, plotter.Ws[-1], plotter.bs[-1], optimizer.dWeights[0], optimizer.dWeights[1], 
                 optimizer.updates[0], optimizer.updates[1], optimizer.previous_loss))
            
        slope.append(optimizer.dlsmooth)
        lossdata.append(optimizer.previous_loss)
        ecolifigwidge.data[0].x=model.inputs
        ecolifigwidge.data[0].y=model.outputs

        ecolifigwidge.data[1].x=model.inputs
        ecolifigwidge.data[1].y=model(model.inputs)
        
        if epoch % 20 == 0 :
            ecolifigwidge.data[2].x=plotter.xs
            ecolifigwidge.data[2].y=plotter.ls

            ecolifigwidge.data[3].x=plotter.xs
            ecolifigwidge.data[3].y=plotter.dWs

            ecolifigwidge.data[4].x=plotter.xs
            ecolifigwidge.data[4].y=plotter.dbs
        
        if optimizer.previous_loss <  1.05:
            counter.set_limit(epoch) 




class RegressionModel(object):

  def __init__(self, weights):
    # Initialize the weights to `5.0` and the bias to `0.0`
    # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
    #weights = np.random.uniform(0,1,2)
    self.W = tf.Variable(weights[0])
    self.b = tf.Variable(weights[1])
    TRUE_W = 3.0
    TRUE_b = 2.0
    NUM_EXAMPLES = 1000
    
    self.inputs  = tf.random.normal(shape=[NUM_EXAMPLES])
    noise   = tf.random.normal(shape=[NUM_EXAMPLES], stddev=1.0)
    self.outputs = self.inputs * TRUE_W + TRUE_b + noise

  def __call__(self, x):
    return self.W * x + self.b

  def update(self, dweights):
      self.W.assign_sub(dweights[0])
      self.b.assign_sub(dweights[1])
     
  def status_string(self):
      status = '{:6.2f} {:6.2f}'.format(self.W.numpy(), self.b.numpy())
      return status
         
  def header_string(self):
      header = '  W      b  '
      return header

  


class RegressionCase(object):

  def __init__(self, model, optimizer):
    self.model = model
    self.optimizer=optimizer
    

  def __call__(self):
    self.model.update(self.optimizer(self.model))
    
    
    

class BaseImage():
    def __init__(self, path):
        self.path = path 

    def row(self, rowi):
     row = self.imageArray[rowi]
     return row

    def getImage(self):
     return self.image 

class GreyImage(BaseImage):
      
  def __init__(self, path):
    super().__init__( path) 
    
  def open(self):
    img = Image.open(self.path)
    self.image = img.convert('L')
    #self.rchannel = self.image.getchannel("R")
    self.imageArray =np.asarray(self.image)
    print (self.image.format, self.image.size, self.image.mode)



def get_optimizer( opt_type, learning_rate):
    opt_type = opt_type.lower()
    
    if opt_type == 'ecoli' :
        optimizer = pctopts.Ecoli(learning_rate)

    if opt_type == 'sgd' :
        optimizer = tf.keras.optimizers.SGD(learning_rate)

    if opt_type == 'adam' :
        optimizer = tf.keras.optimizers.Adam(learning_rate)

    if opt_type == 'adadelta' :
        optimizer = tf.keras.optimizers.Adadelta(learning_rate)

    if opt_type == 'rmsprop' :
        optimizer = tf.keras.optimizers.RMSprop(learning_rate)

    return optimizer
    
def get_metric_type( metrics):
    if len(metrics)>0:        
        if metrics[0] == 'my_mse':
           metrics[0] = pctmetrics.MyMeanSquaredError()
            
        if metrics[0] == 'sse':
           metrics[0] = pctmetrics.SquaredSumError()

        if metrics[0] == 'rsse':
           metrics[0] = pctmetrics.RootSquaredSumError()

    return metrics

def get_loss_function( loss_type):
    loss_name = 'Loss_'+loss_type
    loss_fn=None
    if loss_type == 'my_mse':
        loss_fn = pctloss.MyMeanSquaredError(name=loss_name )
        
    if loss_type == 'sse':
        loss_fn = pctloss.SquaredSumError(name=loss_name )
               
    if loss_type == 'rsse':
        loss_fn = pctloss.RootSquaredSumError(name=loss_name )

    if loss_type == 'rsuse':
        loss_fn = pctloss.RootSumSquaredError(name=loss_name )



    return loss_fn
