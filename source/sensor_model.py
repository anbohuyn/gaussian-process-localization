# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 03:15:40 2017

@author: abhyudai
"""


import numpy as np
import math

class SensorModel:
    
    def __init__(self, gaussian_models):
        self.gaussians = gaussian_models     
        self.num_gaussians = len(gaussian_models)          
        self.sigma = 3.5 #4
        
     #Returns the observation likelihood for all particles as a vector of length = num_particles  
    def get_observation_likelihood(self, particle_locations, observations):        
        n = particle_locations.shape[0]
        observation_likelihood = np.zeros((n,))
        
        observed_values = observations[0][0]
        print "Observed reading", observed_values
 
        #Particle Number i
        for i in range(n):
            location = particle_locations[i,:2]            
            gaussian_likelihood = np.zeros((self.num_gaussians,))
            #Gaussian number g
            for g in range(self.num_gaussians):
                regression_output = self.gaussians[g].predict_gaussian_value(location)
                gaussian_likelihood[g] = math.exp(-np.power(observed_values[g] - regression_output, 2.) / (2 * np.power(self.sigma, 2.)))
                                
            observation_likelihood[i]  =  self.geometric_mean(gaussian_likelihood)           
            if(i%25 == 0):            
                print "Loc", location, "obs", "reg", regression_output, "P(s/x):", observation_likelihood[i]            
    
                        
        return observation_likelihood
        
    def geometric_mean(self, arr):
        N = len(arr)
        value = 1
        for i in range(0,N):
            value*=arr[i]
        return value**(1.0/N)