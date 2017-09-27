# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 17:47:03 2017

@author: abhyudai
"""

#Entry point to run code

from main_controller import MainController

mc = MainController(num_ap = 20)
mc.load_data()
mc.initialize_gaussian_processes(visualize=False)  #Visualize the wifi data in 3D space

#Visualize gaussians after regression
print "Number of gaussians", mc.number_of_access_points
for ap_index in range(0,mc.number_of_access_points):
    pass #mc.visualize_gaussian(ap_index, mc.gaussian_processes[ap_index])
    
#Run particle filter for all gaussians
n_particles = 200
num_iterations = 2000
mc.start_particle_filter(mc.gaussian_processes, mc.wifi_values, num_particles= n_particles, num_iter = num_iterations)
