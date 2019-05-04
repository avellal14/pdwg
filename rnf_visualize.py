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
quality = 0.3
marker_size = 10/2
marker_line = 0.3/2

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
grid_manifold = np.load(exp_dir+'grid_manifold.npy')
rec_data_manifolds = np.load(exp_dir+'rec_data_manifolds.npy')
rec_grid_manifolds = np.load(exp_dir+'rec_grid_manifolds.npy')

############################# TEST ##############################################

def get_full_grid_samples(resolution=100, range_min=-1, range_max=1):
    x0_range = np.linspace(range_min, range_max, 2*resolution+1)
    x1_range = np.linspace(range_min, range_max, 2*resolution+1)
    x0v, x1v = np.meshgrid(x0_range, x1_range)
    grid_flat = np.concatenate([x0v.flatten()[:, np.newaxis], x1v.flatten()[:, np.newaxis]], axis=1)
    grid = np.concatenate([x0v[:,:,np.newaxis], x1v[:,:,np.newaxis]], axis=2)
    return grid_flat, grid, x0_range, x1_range

def get_sparse_grid_samples(resolution=100, subsample_rate=10, range_min=-1, range_max=1):
    full_grid_flat, full_grid, _, _ = get_full_grid_samples(resolution=resolution, range_min=range_min, range_max=range_max)
    index_x0v, index_x1v = np.meshgrid(np.arange(full_grid.shape[0]), np.arange(full_grid.shape[1]))
    index_grid_flat = np.concatenate([index_x0v.flatten()[:, np.newaxis], index_x1v.flatten()[:, np.newaxis]], axis=1)
    index_grid = np.concatenate([index_x0v[:,:,np.newaxis], index_x1v[:,:,np.newaxis]], axis=2)
    subsample_mask = ((index_grid_flat[:, 0]%subsample_rate == 0)+ (index_grid_flat[:, 1]%subsample_rate == 0))>0
    return full_grid_flat[subsample_mask,:]

def obj_fun(X):
    func_scale_1 = 0.5
    func_scale_2 = 0.2
    vals = np.zeros((X.shape[0],))
    for i in range(X.shape[0]):
        vals[i] = func_scale_1*(func_scale_2*X[i,0]**2-0.5*np.cos(2*np.pi*X[i,0])+X[i,1]**2-0.5*np.cos(2*np.pi*X[i,1]))
    return vals


assert(rec_data_manifolds.shape[0] == rec_grid_manifolds.shape[0])
fig, ax = plt.subplots(figsize=(10, 10))
for i in range(rec_data_manifolds.shape[0]):
    plt.clf()
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(data_manifold[:, 0], data_manifold[:, 1], data_manifold[:, 2], s=marker_size, lw = marker_line, edgecolors='k', facecolors=cm.get_cmap("coolwarm")(Normalize()(data_manifold[:, 2])))
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.scatter(rec_data_manifolds[i, :, 0], rec_data_manifolds[i, :, 1], rec_data_manifolds[i, :, 2], s=marker_size, lw = marker_line, edgecolors='k', facecolors=cm.get_cmap("coolwarm")(Normalize()(rec_data_manifolds[i, :, 2])))
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.scatter(grid_manifold[:, 0], grid_manifold[:, 1], grid_manifold[:, 2], s=marker_size, lw = marker_line, edgecolors='k', facecolors=cm.get_cmap("coolwarm")(Normalize()(grid_manifold[:, 2])))
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.scatter(rec_grid_manifolds[i, :, 0], rec_grid_manifolds[i, :, 1], rec_grid_manifolds[i, :, 2], s=marker_size, lw = marker_line, edgecolors='k', facecolors=cm.get_cmap("coolwarm")(Normalize()(rec_grid_manifolds[i, :, 2])))

    fig.canvas.mpl_connect('motion_notify_event', get_on_move(fig, ax1, ax2))
    fig.canvas.mpl_connect('motion_notify_event', get_on_move(fig, ax3, ax4))
    ax1.set_xlim(range_1_min, range_1_max)
    ax1.set_ylim(range_1_min, range_1_max)
    ax1.set_zlim(range_1_min, range_1_max)
    ax2.set_xlim(range_1_min, range_1_max)
    ax2.set_ylim(range_1_min, range_1_max)
    ax2.set_zlim(range_1_min, range_1_max)
    ax3.set_xlim(range_1_min, range_1_max)
    ax3.set_ylim(range_1_min, range_1_max)
    ax3.set_zlim(range_1_min, range_1_max)
    ax4.set_xlim(range_1_min, range_1_max)
    ax4.set_ylim(range_1_min, range_1_max)
    ax4.set_zlim(range_1_min, range_1_max)

    set_axis_prop(ax1, grid_on, ticks_on, axis_on )
    set_axis_prop(ax2, grid_on, ticks_on, axis_on )
    set_axis_prop(ax3, grid_on, ticks_on, axis_on )
    set_axis_prop(ax4, grid_on, ticks_on, axis_on )

    ax1.view_init(elev=25, azim=20)
    ax2.view_init(elev=25, azim=20)
    ax3.view_init(elev=25, azim=20)
    ax4.view_init(elev=25, azim=20)
    plt.draw()
    plt.savefig(exp_dir+'epoch_'+str(i)+'.png', bbox_inches='tight', format='png', dpi=int(quality*my_dpi), transparent=False)
    print('Epoch: ', i)


plt.clf()
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.scatter(data_manifold[:, 0], data_manifold[:, 1], data_manifold[:, 2], s=marker_size, lw = marker_line, edgecolors='k', facecolors=cm.get_cmap("coolwarm")(Normalize()(data_manifold[:, 2])))
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.scatter(rec_data_manifolds[-1, :, 0], rec_data_manifolds[-1, :, 1], rec_data_manifolds[-1, :, 2], s=marker_size, lw = marker_line, edgecolors='k', facecolors=cm.get_cmap("coolwarm")(Normalize()(rec_data_manifolds[-1, :, 2])))
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.scatter(grid_manifold[:, 0], grid_manifold[:, 1], grid_manifold[:, 2], s=marker_size, lw = marker_line, edgecolors='k', facecolors=cm.get_cmap("coolwarm")(Normalize()(grid_manifold[:, 2])))
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.scatter(rec_grid_manifolds[-1, :, 0], rec_grid_manifolds[-1, :, 1], rec_grid_manifolds[-1, :, 2], s=marker_size, lw = marker_line, edgecolors='k', facecolors=cm.get_cmap("coolwarm")(Normalize()(rec_grid_manifolds[-1, :, 2])))

fig.canvas.mpl_connect('motion_notify_event', get_on_move(fig, ax1, ax2))
fig.canvas.mpl_connect('motion_notify_event', get_on_move(fig, ax3, ax4))
ax1.set_xlim(range_1_min, range_1_max)
ax1.set_ylim(range_1_min, range_1_max)
ax1.set_zlim(range_1_min, range_1_max)
ax2.set_xlim(range_1_min, range_1_max)
ax2.set_ylim(range_1_min, range_1_max)
ax2.set_zlim(range_1_min, range_1_max)
ax3.set_xlim(range_1_min, range_1_max)
ax3.set_ylim(range_1_min, range_1_max)
ax3.set_zlim(range_1_min, range_1_max)
ax4.set_xlim(range_1_min, range_1_max)
ax4.set_ylim(range_1_min, range_1_max)
ax4.set_zlim(range_1_min, range_1_max)

set_axis_prop(ax1, grid_on, ticks_on, axis_on )
set_axis_prop(ax2, grid_on, ticks_on, axis_on )
set_axis_prop(ax3, grid_on, ticks_on, axis_on )
set_axis_prop(ax4, grid_on, ticks_on, axis_on )

animate_mode = 'elevazim'
animate([ax1, ax2, ax3, ax4], n_turns = 1, skip_rate = 20, mode=animate_mode, quality=quality, grid_on=grid_on, ticks_on=ticks_on, axis_on=axis_on, save_path=exp_dir+animate_mode+'_rotation')
animate_mode = 'elevation'
animate([ax1, ax2, ax3, ax4], n_turns = 1, skip_rate = 20, mode=animate_mode, quality=quality, grid_on=grid_on, ticks_on=ticks_on, axis_on=axis_on, save_path=exp_dir+animate_mode+'_rotation')
animate_mode = 'azimuth'
animate([ax1, ax2, ax3, ax4], n_turns = 1, skip_rate = 20, mode=animate_mode, quality=quality, grid_on=grid_on, ticks_on=ticks_on, axis_on=axis_on, save_path=exp_dir+animate_mode+'_rotation')
# if not platform.dist()[0] == 'Ubuntu': plt.show()






