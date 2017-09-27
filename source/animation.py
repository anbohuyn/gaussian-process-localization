# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 03:17:01 2017

@author: abhyudai
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import particle_filter_runner as part_filt_anim
#from matplotlib import lines
import time

class ParticleFilterAnimation:
    
    def __init__(self):
        self.pf_runner = None
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-2, 28)
        self.ax.set_ylim(-20, 20)
        self.ax.axis('equal')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Wifi localization')
        self.xdata = []
        self.ydata = []
        self.plt_handle, = plt.plot([], [], 'g', markersize=2, marker=".", animated=True)
        self.num_iter = 2000 # Total iterations are 1400
            
    def initialize(self):        
        im = plt.imread('../data/UCM_data/RealRobot/map.png')
        plt.imshow(im, extent=[-20,20,-3,27])

#        for i in range(0,len(self.walls)):
#            wall = self.walls[i]      
#            line_x = [ [wall[0],wall[2]] ]
#            line_y = [ [wall[1],wall[3]] ]
#            line = lines.Line2D(line_x, line_y, zorder = 0)
#            self.ax.add_line(line)
#            
        return self.plt_handle,
                    
    def update(self, frame):
        #NOTE flipped
        self.xdata = -frame[:,1]  #.append(frame[:,0])
        self.ydata = frame[:,0] #.append(frame[:,1])
        self.plt_handle.set_data(self.xdata, self.ydata)
        return self.plt_handle,
    
    def plot_particles(self, particleFilter):
        FuncAnimation(self.fig, self.update, frames=particleFilter.data_gen, init_func=self.initialize,interval=25, blit=True)
        plt.show()
    
    def save_animation(self, particleFilter):
        ani = FuncAnimation(self.fig, self.update, frames=particleFilter.data_gen, init_func=self.initialize, blit=True, repeat = False, save_count=self.num_iter)
        ani.save('../output/robot_animation_'+time.strftime("%Y%m%d-%H%M%S")+'.mp4', writer ='ffmpeg')

    def start_particle_filter(self, num_particles, gaussian_models, log_data, save_mode, num_iter):
        if num_iter > 0:        
            self.num_iter = num_iter
        self.pf_runner = part_filt_anim.ParticleFilterRunner(num_particles, gaussian_models, log_data)
        if save_mode:
            self.save_animation(self.pf_runner)    
        else:
            self.plot_particles(self.pf_runner)
