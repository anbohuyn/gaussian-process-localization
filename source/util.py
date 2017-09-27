# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 01:25:03 2017

@author: abhyudai
"""

import numpy as np

class TimeUtil:
    
    def __init__(self, time_threshold):
        self.time_threshold = time_threshold
    
    #Bring all readings to a common ground
    def round_time(self, t):
        return int(t/self.time_threshold)*self.time_threshold
    
    
class MathUtil:
    
    def sample_from_normal(self, mean, variance):
        if(abs(variance) == 0):
            return mean
        else:
            return np.random.normal(mean, variance)