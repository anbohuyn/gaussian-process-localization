# -*- coding: utf-8 -*-
"""
Created on Mon May  1 02:00:15 2017

@author: abhyudai
"""

import gaussian
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 

class MainController:
    
    def __init__(self, num_ap):    
        #Close any old plots
        plt.close("all")
        self.number_of_access_points = num_ap   
        self.import_dir = '../data/UCM_data/generated/'
        self.gaussian_processes = [None]*num_ap
        
    #import generated data files (run read_UCM_data.py)    
    def load_data(self):        
        self.odometry = np.load(self.import_dir+'odometry.npy')
        self.odometry_full = np.load(self.import_dir+'odometry_complete.npy')
        self.wifi_timestamps = np.load(self.import_dir+'wifi_timestamps.npy')
        self.wifi_rssi = np.load(self.import_dir+'wifi_rssi.npy')
        self.wifi_locations = np.load(self.import_dir+'wifi_locations.npy')        
        self.wifi_values = np.zeros((len(self.wifi_locations),self.number_of_access_points))
 
   #Visualize wifi data       
    def visualize_wifi_data(self, wifi_locations, wifi_values, ap):
        visualize = True
        if(visualize):
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_zlim([-100.0,0])
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            ax.set_zlabel('Wifi Signal Strength(dB)')
            ax.set_title('Wifi measurements for AP # ' + str(ap))
            ax.scatter(wifi_locations[:,0], wifi_locations[:,1], wifi_values, c='orange')
        
    #Get priors on wifi data
    def preprocess_wifi_data(self,  wifi_locations, wifi_rssi, ap):               
        wifi_rssi_mean = np.mean(wifi_rssi, axis=2)
        wifi_rssi_cov = np.std(wifi_rssi, axis = 2)
        return (wifi_rssi_mean[:,ap], wifi_rssi_cov[:,ap])
        
    #Initialize GPs
    def initialize_gaussian_processes_optimal(self, visualize=True):
        gaussians = []
        for i in np.arange(0,self.number_of_access_points,1):
            # There are multiple APs, need to create multiple GPs
            print "Gaussian Process #", i
            gp = gaussian.GaussianProcess()
            wifi_rssi_mean_ap, wifi_rssi_cov_ap = self.preprocess_wifi_data(self.wifi_locations, self.wifi_rssi, i)
            if(visualize == True):
                self.visualize_wifi_data(self.wifi_locations, wifi_rssi_mean_ap,i)
            gp.set_param_ranges(i)      
            gp.train_gaussian_model(self.wifi_locations, wifi_rssi_mean_ap)
            gp.set_prior_covariance(wifi_rssi_cov_ap)
            gaussians.append(gp)
            self.wifi_values[:,i] = wifi_rssi_mean_ap            
        self.gaussian_processes = gaussians
        
    #Initialize GPs using preset parameters
    def initialize_gaussian_processes(self, visualize=True):
        gaussians = []
        for i in np.arange(0,self.number_of_access_points,1):
            # There are multiple APs, need to create multiple GPs
            print "Gaussian Process #", i
            gp = gaussian.GaussianProcess()
            wifi_rssi_mean_ap, wifi_rssi_cov_ap = self.preprocess_wifi_data(self.wifi_locations, self.wifi_rssi, i)
            if(visualize == True):
                self.visualize_wifi_data(self.wifi_locations, wifi_rssi_mean_ap,i)
            gp.train_gaussian_model_with_params(self.wifi_locations, wifi_rssi_mean_ap, self.default_params())
            gp.set_prior_covariance(wifi_rssi_cov_ap)
            gaussians.append(gp)
            self.wifi_values[:,i] = wifi_rssi_mean_ap 
        self.gaussian_processes = gaussians

    #Saved parameter rang values    
    def default_params(self):
        return [8,2,0.5]
                      
    #Test code for mean offset
    def test_mean_offset(self, gp):
        test_x = np.zeros((4,2))
        test_x[0,:]= [4,-15]
        test_x[1,:]= [0,-20]
        test_x[2,:]= [18,2]
        test_x[3,:]= [100,100] 
        for t in range(len(test_x)):
            print test_x[t,:], gp.predict_mean_offset(test_x[t,:])
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Wifi Signal Mean(dB)')
        ax.set_title('Wifi mean analyzer')
        X_range = np.arange(0,20,1)
        Y_range = np.arange(-22,20,1)
        for i in range(0,len(X_range)):
            for j in range(0,len(Y_range)):
                x = X_range[i]
                y = Y_range[j]
                t = np.array([x,y])
                t.reshape(1,2)        
                z = gp.predict_mean_offset(t)
                ax.scatter(x,y,z, c='green')
        
        for i in range(len(test_x)):
            t = test_x[i,:]
            k = gp.build_vector_covariance(t)
            print t, np.sum(k)

    #Prediction using gaussian, extrapolate for entire grid
    def visualize_gaussian(self, ap_index, gp):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Wifi Signal Strength(dB)')
        ax.set_title('Gaussian fit for AP # ' + str(ap_index))
        ax.set_zlim([-100.0,0])
        
        X_range = np.arange(0,20,1)
        Y_range = np.arange(-22,20,1)
        Y,X = np.meshgrid(Y_range,X_range)
        Z=np.zeros(shape = X.shape)
        
        for i in range(0,len(X_range)):
            for j in range(0,len(Y_range)):
                x = X_range[i]
                y = Y_range[j]
                Z[i,j] = gp.predict_gaussian_value([x,y])
                        
        ax.plot_surface(X,Y,Z)
        plt.show()

    #Let's begin the particle filter
    def start_particle_filter(self,  gaussians, wifi_values, num_particles, num_iter):
        import animation
        particle_filter_animation = animation.ParticleFilterAnimation()
        
        wifi_log = np.zeros((len(self.wifi_timestamps ),1+self.number_of_access_points))
        wifi_log[:,0] = self.wifi_timestamps [:,0]
        wifi_log[:,1:] = np.array(wifi_values)
        
        log_data = {'odometry':self.odometry_full, 'wifi': wifi_log}
        particle_filter_animation.start_particle_filter(num_particles = num_particles, gaussian_models = gaussians, log_data=log_data, save_mode = True, num_iter=num_iter)