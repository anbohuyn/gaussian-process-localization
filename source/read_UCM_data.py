# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:01:56 2017

@author: abhyudai
"""

import scipy.io as sio
import numpy as np
import os,errno

parent_dir = '../data/UCM_data/'
save_dir = parent_dir+'generated/'

if not os.path.exists(os.path.dirname(save_dir)):
    try:
        os.makedirs(os.path.dirname(save_dir))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

matfile = sio.loadmat(parent_dir + 'RealRobot_data.mat')
locations = matfile['locs']
data = matfile['data']

#Processed data
RealRobot_time = sio.loadmat(parent_dir + 'RealRobot_time.mat')
wifi_timestamps = RealRobot_time['wifiTimeStamps'] #110x1

#Runs
num_runs = 10
wifi_runs = []

for i in range(10):
    run = str(i+1)
    if len(run) == 1:
        run = "0"+run
        
    run_mat = sio.loadmat(parent_dir + 'Runs/'+run+'.mat')
    wifi_readings = run_mat['classificationData']
    location_data = run_mat['classificationLabels']
    
    wifi_runs.append(wifi_readings)
    #loc_runs.append(location_data)  #Location data does not change

wifi_combined = np.array(wifi_runs) #Shape is 10,110,48
wifi_combined = np.swapaxes(wifi_combined,0,1)
wifi_combined = np.swapaxes(wifi_combined,1,2) #Shape is now 110,48,10

location_points = np.multiply(location_data,0.3048)
#This is same as ground truth..

np.save(save_dir+'wifi_rssi.npy', wifi_combined)
np.save(save_dir+'wifi_locations', location_points)
np.save(save_dir+'wifi_timestamps', wifi_timestamps)

a = np.multiply(location_points[:,1],-1)
b = location_points[:,0]
location_flipped = np.transpose(np.vstack((a,b)))

#Odometry Data

odo_data = np.genfromtxt(parent_dir +'RealRobot/dImage.txt')
filtered_odometry = odo_data[odo_data[:,5] > 0][:,1:5]
test_odomotery = odo_data[odo_data[:,5] == 0][:,1:5]
full_odomotery = odo_data[:,1:5]

import matplotlib.pyplot as plt
plt.figure()
plt.title("Raw odometry data")

#Flip x and y
#Flip x around origin
x_val = np.multiply(filtered_odometry[:,2],-1)
y_val = filtered_odometry[:,1]
plt.plot(x_val, y_val)


#Ground truth data

gt_raw = np.genfromtxt(parent_dir +'RealRobot/scanLocationsGT.txt', skip_header=1, deletechars=';')
gt_filtered = gt_raw[:,1:7]

converter = np.transpose([0.3048, 0.0254, 0.0015875])
gt_x = np.dot(gt_filtered[:,0:3], converter)
gt_y = np.dot(gt_filtered[:,3:6], converter)

#Flip x and y
#Flip x around origin
x_val_gt = np.multiply(gt_y,-1)
y_val_gt = gt_x

plt.figure()
plt.set_xlim(-2, 28)
plt.set_ylim(-20, 20)
plt.axis('equal')
plt.set_xlabel('X (m)')
plt.set_ylabel('Y (m)')
        
plt.title("Ground truth")
plt.plot(x_val_gt, y_val_gt)

np.save(save_dir+'odometry.npy', filtered_odometry)
np.save(save_dir+'odometry_complete.npy', full_odomotery)
