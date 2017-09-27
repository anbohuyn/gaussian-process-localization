# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 01:36:53 2017

@author: abhyudai
"""

import particle_filter
import numpy as np
import traceback

class ParticleFilterRunner:
            
    def __init__(self, num_particles, gaussian_models, log_data):
        self.num_particles = num_particles
        self.gaussians = gaussian_models
        self.log = log_data
            
    def data_gen(self):
        try:
            particlefilter = particle_filter.ParticleFilter(self.num_particles, self.gaussians)
            particlefilter.initialize_particles()
            
            log = self.log
            odometry = log['odometry']
            observations = log['wifi']
            timestamps = odometry[:,0]
            
            number_of_actions = len(timestamps)
            bot_is_moving = True
            
            prev_location = odometry[0,1:]
            
            print "Number of actions",number_of_actions              
            
            for iteration in range(0,number_of_actions):
                t = timestamps[iteration]
                
                odom_iter = np.argwhere(odometry[:,0]==t)[0][0] 
                curr_location = odometry[odom_iter,1:] 
                odom = curr_location - prev_location  
                prev_location = curr_location
                print "Iter", iteration, "Time:", t, "Odom", odom

                # Odometry
                bot_is_moving = np.any(odom)
                if bot_is_moving:
                    print "Motion Model - Odom:", odom      
                    particlefilter.predict_step(odom)                                         

                sensor_reading_present = np.argwhere(observations[:,0]==t)
                
                if sensor_reading_present.shape[0] > 0 and bot_is_moving: 
                    observation = observations[sensor_reading_present,1:]
                    print "Sensor Update - Reading:", observation
                    particlefilter.update_sensor_model(observation)
                    particlefilter.resample()
                             
                # Plot Particles
                yield particlefilter.particle_location
                particlefilter.compute_centroid()
                   
        except:
            print "Error in main loop", traceback.format_exc()