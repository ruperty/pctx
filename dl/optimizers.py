# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:22:55 2020

@author: rupert
"""

import tensorflow as tf
import numpy as np
from pct.utilities.rmath import sigmoid
from pct.utilities.rmath import smooth
from tensorflow.python.framework import dtypes
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.util.tf_export import keras_export



class EcoliContinuous(object):

  def __init__(self, weights, learning_rate, lossfn, ref_slope=-10,  smooth=0.75, sigmoid_range=5, sigmoid_scale=0.1):
    self.previous_loss = -1
    self.initial_loss=None
    self.ref_slope=ref_slope
    self.initial_ref_slope=ref_slope
    self.tumble_threshold=-25
    self.initial_tumble_threshold=self.tumble_threshold
    self.nweights=weights
    self.dWeights = np.random.uniform(-1,1,self.nweights)
    self.lossfn=lossfn
    self.learning_rate = learning_rate
    self.dl=None
    self.dlsmooth=None
    self.dlsmoothfactor=smooth
    self.updates=np.zeros(weights)
    #self.period=period
    #self.period_loss_sum=0
    #self.previous_historical_loss =-1
    self.sigmoid_range=sigmoid_range
    self.sigmoid_scale=sigmoid_scale
    self.slope_error_accumulator=self.tumble_threshold

  def __call__(self, model):
    current_loss = self.lossfn(model.outputs, model(model.inputs))
    if self.initial_loss == None:
        self.initial_loss = current_loss
    
    #print(current_loss.numpy())
    
    if self.previous_loss > 0:
        self.current_slope=current_loss-self.previous_loss
    else:
        self.current_slope=0
        
    self.slope_error=self.ref_slope-self.current_slope
    
    if self.slope_error < 0:
        self.slope_error_accumulator += self.slope_error
    
    if self.slope_error_accumulator < self.tumble_threshold  :
        self.dWeights = np.random.uniform(-1,1,self.nweights)            
        self.updates = sigmoid( current_loss *self.learning_rate * self.dWeights, self.sigmoid_range, self.sigmoid_scale)
        self.slope_error_accumulator=0
        if current_loss< self.initial_loss:
            self.ref_slope = self.initial_ref_slope * current_loss / self.initial_loss
            self.tumble_threshold = self.initial_tumble_threshold * current_loss / self.initial_loss
                        
    if self.previous_loss >= 0:
        if self.dlsmooth==None:
            self.dlsmooth = self.current_slope
        else:
            self.dlsmooth = smooth( self.current_slope, self.dlsmooth, self.dlsmoothfactor)       
            
    self.previous_loss = current_loss

    return self.updates

  def status_string(self):
      
      wts = []
      for i in range(self.nweights):
          wts.append( '{0:6.2f}'.format(self.dWeights[i])  )
          
      ups = []
      for i in range(self.nweights):
          ups.append( '{0:6.2f}'.format(self.updates[i])  )
          
          
      #wts_str = join(wts)
      status = '{} {} {:8.2f} {:8.2f} {:8.2f} {:8.2f} {:8.2f} {:6.2f}'.format(''.join(wts), ''.join(ups),
               self.previous_loss, self.ref_slope, self.current_slope, self.slope_error, 
               self.slope_error_accumulator, self.tumble_threshold)
      
      return status
  
  def header_string(self):
      header = '  dW    db     dWc   dbc    loss     ref      perc      err      acc'
      return header


  

class EcoliPeriodic(object):

  def __init__(self, weights, learning_rate, lossfn, smooth=0.75, period=1, sigmoid_range=5, sigmoid_scale=0.1):
    self.previous_loss = -1
    self.nweights=weights
    self.dWeights = np.random.uniform(-1,1,self.nweights)
    self.lossfn=lossfn
    self.learning_rate = learning_rate
    self.dl=None
    self.dlsmooth=None
    self.dlsmoothfactor=smooth
    self.updates=np.zeros(weights)
    self.period=period
    self.ctr=1
    self.period_loss_sum=0
    self.previous_historical_loss =-1
    self.sigmoid_range=sigmoid_range
    self.sigmoid_scale=sigmoid_scale

  def __call__(self, model):
    current_loss = self.lossfn(model.outputs, model(model.inputs))
    #print(current_loss.numpy())
    self.add_period_loss(current_loss)
    
    if self.ctr % self.period ==0:
        self.current_historical_loss = self.get_mean_loss()
        #print("m", self.current_historical_loss.numpy(), self.previous_historical_loss)
        if self.current_historical_loss >= self.previous_historical_loss  or self.previous_historical_loss <0 :
            self.dWeights = np.random.uniform(-1,1,self.nweights)
            
            
            self.updates = sigmoid( self.current_historical_loss  *self.learning_rate * self.dWeights, self.sigmoid_range, self.sigmoid_scale)
            
            #self.updates = [sigmoid( self.current_historical_loss  *self.learning_rate * self.dWeights[0], self.sigmoid_range, self.sigmoid_scale), 
             #       sigmoid( self.current_historical_loss  *self.learning_rate * self.dWeights[1], self.sigmoid_range, self.sigmoid_scale)]
            
        if self.previous_historical_loss >= 0:
            self.dl = self.current_historical_loss-self.previous_historical_loss
            if self.dlsmooth==None:
                self.dlsmooth = self.dl
            else:
                self.dlsmooth = smooth( self.dl, self.dlsmooth, self.dlsmoothfactor)

        self.previous_historical_loss = self.current_historical_loss
    #else:
    #    self.updates=np.zeros(self.nweights)
        
            
    self.previous_loss = current_loss

    self.ctr+=1
    return self.updates

  def add_period_loss(self, loss):
    self.period_loss_sum+=loss        
    
  def get_mean_loss(self):
      self.mean_loss = self.period_loss_sum/self.period
      self.period_loss_sum=0
      return self.mean_loss 



@keras_export("keras.optimizers.Ecoli")
class Ecoli(optimizer_v2.OptimizerV2):
  """Ecoli optimizer.

  Computes:
  ```
  if loss has increased
  theta(t+1) = theta(t) - learning_rate * loss * random_gradient
  ```

  # References
      Powers, WT
      http://www.livingcontrolsystems.com/demos/ecoli/ecoli.pdf
  """

  def __init__(self,
               learning_rate=0.01,
               name="Ecoli",
               **kwargs):
    """Construct a new Ecoli optimizer.

    Arguments:
      learning_rate: float hyperparameter >= 0. Learning rate.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to 'Ecoli'.
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
    """
    super(Ecoli, self).__init__(name, **kwargs)
    self.loss_historical=tf.Variable(0.0)
    self.learning_rate=learning_rate

  def minimize(self, loss, var_list, grad_loss=None, name=None):
        grads_and_vars = self._compute_gradients(
            loss, var_list=var_list, grad_loss=grad_loss)
    
        return self.apply_gradients(grads_and_vars, name=name)

  def _resource_apply_dense(self, grad, var, apply_state=None):
      #print(grad.numpy())
      #self.minimize( loss, var):
      tf.print(grad)      
      tf.print(var)      
      return apply_ecoli(var, self.learning_rate, grad, self.loss_historical)

  def _compute_gradients(self, loss, var_list, grad_loss=None):
        loss_value = loss()
        print("ls", loss_value)
        print("hlv", self.loss_historical)
        print("wts", var_list)
        
        if loss_value >= self.loss_historical:
            grads = tf.random.uniform(shape=(len(var_list),), minval=-1., maxval=1.)
            
        self.loss_historical.assign(loss_value)
        
        print("dws", grads)
        #var_list += grads
        grads_and_vars = list(zip(grads, var_list))
        self._assert_valid_dtypes([
            v for g, v in grads_and_vars
            if g is not None and v.dtype != dtypes.resource
        ])
    
        return grads_and_vars

  def _create_slots(self, var_list):
    pass #x=0 # do nothing



  def get_config(self):
    config = super(Ecoli, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
    })
    return config


def apply_ecoli(var, lr, grad, loss):
    result = var + lr * grad * loss
    return result
    