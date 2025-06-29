import numpy as np
import tensorflow as tf
import math

class custom_loss(tf.keras.losses.Loss):
    # initialize instance attributes
    def __init__(self, mean_pr, std_pr):
        super(custom_loss, self).__init__()
        self.mean_pr=mean_pr
        self.std_pr=std_pr
        
    # Compute loss
    def call(self, y_true, y_pred):
        # use weighted MAE for precipitation channel
        pr_org=y_true[:,:,:,5]*self.std_pr+self.mean_pr
        weights= tf.clip_by_value(pr_org/6.0, 0.1, 1.0)
        pr_loss=tf.reduce_mean(tf.multiply(weights, tf.abs(tf.subtract(y_pred[:,:,:,5], y_true[:,:,:,5]))))       
        return tf.reduce_mean(tf.abs(y_true[:,:,:,0:5] - y_pred[:,:,:,0:5]))+pr_loss
