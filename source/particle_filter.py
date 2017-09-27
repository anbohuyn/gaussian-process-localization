# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:06:51 2017

@author: abhyudai
"""

 
import sensor_model as sm       
import util
import numpy as np
mathUtil = util.MathUtil()
import math

class ParticleFilter:
         
    def __init__(self, num_particles, gaussian_models):
        self.sensor_model = sm.SensorModel(gaussian_models)
        self.number_of_particles = num_particles #100
        self.particle_location = np.zeros((self.number_of_particles,3)) 
        self.particle_observation_likelihood = np.zeros((self.number_of_particles)) 
        self.best_particle = None
        self.centroid = None
        self.motion_model_variance = [0.02, 0.02, 0.005]   #0.05,0.05,0.01
        self.raw_location = [0,0,0]
        
    def initialize_particles(self):        
        print "Initializing particle filter..."        
        for particle_index in range(0, len(self.particle_location)):
            self.particle_location[particle_index,:] = [0,0,0]
        print "Number of particles = ", self.number_of_particles        
        
    def reinitialize_from_middle(self, old_location, spread):
        print "Restarting from location",old_location
        for particle_index in range(0,len(self.particle_location)):
            x = mathUtil.sample_from_normal(old_location[0], spread[0]/2) 
            y = mathUtil.sample_from_normal(old_location[1], spread[1]/2) 
            theta = mathUtil.sample_from_normal(old_location[2], 0.05)             
            theta = theta%(2*math.pi)
            new_location = [x,y,theta]
            self.particle_location[particle_index,:] = new_location
            print "Particle # ", particle_index, new_location
            
    def reinitialize(self, new_location):
        print "Initialize Location to",new_location
        for particle_index in range(0,len(self.particle_location)):
            self.particle_location[particle_index,:] = new_location
            
    def predict_step(self, odometry):
        for particle_index in range(0,self.number_of_particles):            
            particle = self.particle_location[particle_index]
            if(particle_index %25 == 0):
                self.particle_location[particle_index] = self.predict_location(particle, odometry, verbose=False)
            else:
                self.particle_location[particle_index] = self.predict_location(particle, odometry, verbose=False)
            #What if no noise, raw_location is based just on odometry
            self.raw_location =  np.add(self.raw_location, odometry) 
            
    def predict_location(self, particle, odometry, verbose):   
        [del_x,del_y,del_theta] = odometry       
        
        [a,b,c] = self.motion_model_variance #self.get_variance_from_odometry(odometry)
        
        x2 = mathUtil.sample_from_normal(del_x, a)    
        y2 = mathUtil.sample_from_normal(del_y, b)        
        theta2 = mathUtil.sample_from_normal(del_theta, c)
        theta2 = theta2%(2*math.pi)
    
        noisy_odometry = [x2,y2,theta2]        

        predicted_location = np.add(particle, noisy_odometry)
        predicted_location[2] = predicted_location[2]%(2*math.pi)        
        
        if(verbose):
            print "Odometry", odometry, "noisy", noisy_odometry
       
        return predicted_location        
                
        
    def normalize_observation_likelihood(self):
        total_obs_likelihood = sum(self.particle_observation_likelihood)
        if(total_obs_likelihood == 0):
            #raise ValueError("Total Observation likelihood is zero, reset to uniform")            
            self.particle_observation_likelihood = [1.0/self.number_of_particles for i in range(0,self.number_of_particles)]
        else:                         
            #print "Obs Lik for all particles before norm:\n", self.particle_observation_likelihood
            self.particle_observation_likelihood = [x/total_obs_likelihood for x in self.particle_observation_likelihood]
            #print "Obs Lik for all particles: after norm", sum(self.particle_observation_likelihood)
    
    def update_sensor_model(self, observation):
        self.particle_observation_likelihood = self.sensor_model.get_observation_likelihood(self.particle_location, observation)
        self.normalize_observation_likelihood()
       
    def calculate_particle_bounds(self):
        min_bounds = np.amin(self.particle_location, axis = 0)
        max_bounds = np.amax(self.particle_location, axis = 0)         
        return np.concatenate((min_bounds[0:2],max_bounds[0:2]))
        
    def resample(self):
        self.best_particle = np.array(self.particle_observation_likelihood).argmax()
        print "Best particle #", self.best_particle, self.particle_location[self.best_particle], "P(max)", self.particle_observation_likelihood[self.best_particle]
        
        new_locations = np.zeros(self.particle_location.shape)
        probabilities = np.asarray(self.particle_observation_likelihood)
        indices = np.arange(self.number_of_particles)
        for i in range(0,self.number_of_particles):        
            new_locations[i] = self.particle_location[np.random.choice(indices, p=probabilities)]    
        self.particle_location = new_locations
        self.particle_observation_likelihood = [1.0/self.number_of_particles for i in range(0,self.number_of_particles)]
    
    def compute_centroid(self):
        self.centroid = [ np.mean(self.particle_location[:,i]) for i in range(0,3)]
        print "Centroid of particles", self.centroid