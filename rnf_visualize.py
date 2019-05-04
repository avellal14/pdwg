"""Random variable transformation classes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pdb 
import numpy as np
import math 
import os
from pathlib import Path
import platform

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 16
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16
my_dpi = 350
grid_on, ticks_on, axis_on = False, False, True
quality=0.3

def get_on_move(fig, ax1, ax2):
    def on_move(event):
        if event.inaxes == ax1:
            if ax1.button_pressed in ax1._rotate_btn:
                ax2.view_init(elev=ax1.elev, azim=ax1.azim)
            elif ax1.button_pressed in ax1._zoom_btn:
                ax2.set_xlim3d(ax1.get_xlim3d())
                ax2.set_ylim3d(ax1.get_ylim3d())
                ax2.set_zlim3d(ax1.get_zlim3d())
        elif event.inaxes == ax2:
            if ax2.button_pressed in ax2._rotate_btn:
                ax1.view_init(elev=ax2.elev, azim=ax2.azim)
            elif ax2.button_pressed in ax2._zoom_btn:
                ax1.set_xlim3d(ax2.get_xlim3d())
                ax1.set_ylim3d(ax2.get_ylim3d())
                ax1.set_zlim3d(ax2.get_zlim3d())
        else:
            return
        fig.canvas.draw_idle()
    return on_move

def set_axis_prop(ax, grid_on, ticks_on, axis_on):
    ax.grid(grid_on)
    if not ticks_on:
        if hasattr(ax, 'set_xticks'): ax.set_xticks([])
        if hasattr(ax, 'set_yticks'): ax.set_yticks([])
        if hasattr(ax, 'set_zticks'): ax.set_zticks([])
    if not axis_on: ax.axis('off')

def animate(ax_list, n_turns = 1, skip_rate = 10, mode='elevazim', wait_time=0.05, save_path=None, quality=0.3, grid_on=False, ticks_on=False, axis_on=True):
    if save_path is None: 
        plt.show(block=False)
        for i in range(360*n_turns):
            if i % skip_rate == 0:
                if mode == 'azimuth': 
                	for ax in ax_list: ax.view_init(elev=10., azim=i)
                elif mode == 'elevation':
                    for ax in ax_list: ax.view_init(elev=i, azim=10)
                if mode == 'elevazim': 
                    for ax in ax_list: ax.view_init(elev=i, azim=i)
                else: pdb.set_trace()
                for ax in ax_list: set_axis_prop(ax, grid_on, ticks_on, axis_on)
                plt.draw()
                print('(Elev, Azim): ', ax_list[0].elev, ax_list[0].azim)
                plt.pause(wait_time)
    else:
        for i in range(360*n_turns):
            if i % skip_rate == 0:
                if mode == 'azimuth': 
                    for ax in ax_list: ax.view_init(elev=10., azim=i)
                elif mode == 'elevation':
                    for ax in ax_list: ax.view_init(elev=i, azim=10)
                if mode == 'elevazim': 
                    for ax in ax_list: ax.view_init(elev=i, azim=i)
                else: pdb.set_trace()
                for ax in ax_list: set_axis_prop(ax, grid_on, ticks_on, axis_on)
                plt.savefig(save_path+'_'+str(i)+'.png', bbox_inches='tight', format='png', dpi=int(quality*my_dpi), transparent=False)
                print('(Elev, Azim): ', ax_list[0].elev, ax_list[0].azim)


range_1_min = -1
range_1_max = 1

exp_dir = str(Path.home())+'/ExperimentalResults/RNF_EXP/'
data_manifold = np.load(exp_dir+'data_manifold.npy')
rec_data_manifolds = np.load(exp_dir+'rec_data_manifolds.npy')

fig, ax = plt.subplots(figsize=(10, 5))
# plt.show(block=False)
for i in range(rec_data_manifolds.shape[0]):
	plt.clf()
	ax1 = fig.add_subplot(1, 2, 1, projection='3d')
	ax1.scatter(data_manifold[:, 0], data_manifold[:, 1], data_manifold[:, 2], facecolors=cm.get_cmap("coolwarm")(Normalize()(data_manifold[:, 2])))
	ax2 = fig.add_subplot(1, 2, 2, projection='3d')
	ax2.scatter(rec_data_manifolds[i, :, 0], rec_data_manifolds[i, :, 1], rec_data_manifolds[i, :, 2], facecolors=cm.get_cmap("coolwarm")(Normalize()(rec_data_manifolds[i, :, 2])))

	c1 = fig.canvas.mpl_connect('motion_notify_event', get_on_move(fig, ax1, ax2))
	ax1.set_zlim(range_1_min, range_1_max)
	ax2.set_zlim(range_1_min, range_1_max)
	set_axis_prop(ax, grid_on, ticks_on, axis_on )
	
	plt.draw()
	# plt.pause(0.1)
	set_axis_prop(ax1, grid_on, ticks_on, axis_on )
	set_axis_prop(ax2, grid_on, ticks_on, axis_on )
	plt.savefig(exp_dir+'epoch_'+str(i)+'.png', bbox_inches='tight', format='png', dpi=int(quality*my_dpi), transparent=False)
	print('Epoch: ', i)

plt.clf()
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(data_manifold[:, 0], data_manifold[:, 1], data_manifold[:, 2], facecolors=cm.get_cmap("coolwarm")(Normalize()(data_manifold[:, 2])))
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(rec_data_manifolds[-1, :, 0], rec_data_manifolds[-1, :, 1], rec_data_manifolds[-1, :, 2], facecolors=cm.get_cmap("coolwarm")(Normalize()(rec_data_manifolds[-1, :, 2])))

c1 = fig.canvas.mpl_connect('motion_notify_event', get_on_move(fig, ax1, ax2))
ax1.set_zlim(range_1_min, range_1_max)
ax2.set_zlim(range_1_min, range_1_max)
set_axis_prop(ax, grid_on, ticks_on, axis_on )

animate_mode = 'elevazim'
animate([ax1, ax2], n_turns = 1, skip_rate = 20, mode=animate_mode, quality=quality, grid_on=grid_on, ticks_on=ticks_on, axis_on=axis_on, save_path=exp_dir+animate_mode+'_rotation')

if not platform.dist()[0] == 'Ubuntu': plt.show()






