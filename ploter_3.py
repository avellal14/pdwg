import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import pdb
import scipy 
from os import listdir
import os
import os.path
import glob
from os.path import isfile, join
import subprocess
import re

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
# plt.rcParams['axes.titlesize'] = 10
# plt.rcParams['legend.fontweight'] = 'normal'
# plt.rcParams['figure.titlesize'] = 12

my_dpi = 200
fig_width = 9*200
fig_height = 6*200

# exp_folders = ['./experiments-Det-WAE-CELEB/15e26667130f427eb2ff4d402a85d3dc/',
#                './experiments-Det-WGANGP-CELEB/53ec8ff3559f40ddb8db6f9d6fd7aeea/',
#                './experiments-Det-PDWGAN-CELEB/0539390c52ce4a518385e957cd269641/']

# exp_folders = ['./experiments-Det-WAE-CIFAR10/9511cff568144e19a8bd68a9d7f6715a/',
#                './experiments-Det-WGANGP-CIFAR10/e5742de883f6452e9a3a4960429ab391/',
#                './experiments-Det-PDWGAN-CIFAR10/4d08f5269c12452495a2e0d2089d22e5/']

# exp_folders = ['./experiments-Det-WAE-FLOWERS/b1806193edef4ef396264cdf8ca0d432/',
#                './experiments-Det-WGANGP-FLOWERS/1909f4b9c7084c1ba04eebed71b49cd1/',
#                # './experiments-Det-PDWGAN-FLOWERS/c411520907d849c5a310a2962e0eff15/']
#                './experiments-Det-PDWGAN-FLOWERS/005eb4cef7e949e08ce48acfc3f46be3/']

# exp_folders = ['./OlderExperiments/fourthSet/experiments-WAE-CIFAR10/60322e3ddc1b4c2d99d53b933e95543b/',
#                './OlderExperiments/fourthSet/experiments-WGANGP-CIFAR10/dfc7daabc5ed44abb88228e7d76b1083/',
#                './OlderExperiments/fourthSet/experiments-PDWGAN-CIFAR10/9de19b3af7c5444bb8dc1fea7b387820/']

# exp_folders = ['./experiments-Det-WAE-CUB/c2733e89919f4baf8aa29064187ee428/',]

# exp_folders = ['./OlderExperiments/fifthSet/experiments128-WAEVanilla-CIFAR10/107c24e9367f490fa5ffa2f2a49efb41/',
#                './experimentsLast-WGANGPCannon-CIFAR10/41f0e373518647429e5df8d6d690c819/',
#                './experimentsLast-PDWGANCannon-CIFAR10/aa525987da2d4feea0e7964b91530240/']

# exp_folders = ['./experimentsLast-WGANGPCannon-CIFAR10/41f0e373518647429e5df8d6d690c819/',
#                './experimentsLast-PDWGANCannon-CIFAR10/aa525987da2d4feea0e7964b91530240/']

exp_folders = ['./EEEexperimentsLast-WGANGPCannon-FLOWERS/44cc2a6852f94b58886539dd3db7c5ac/',
               './EEEexperimentsLast-PDWGANCannon-FLOWERS/6f403789c7df4571bc52dc28238be0bb/']

# exp_folders = ['./EEEexperimentsLast-WGANGPCannon-CUB/b1cd63a4e204451eab898eceb83fc3f4/',
#                './EEEexperimentsLast-PDWGANCannon-CUB/ce3b05f0d1b44c81b6d01319f101c46f/']

save_index = -1
# mode = 'LeastEpochs'
# mode = 'Full'
mode = 'EpochInterval'
min_epoch = 200
max_epoch = 1500
# min_epoch = 0
# max_epoch = 2300
names = ['Epochs', 'FID', 'Model Inc Mean', 'Model Inc Std', 'Real Inc Mean', 'Real Inc Std']
list_of_np_numerics = []
for exp_folder in exp_folders:
    file_path = exp_folder+'/test_inception_stats.txt'

    if os.path.exists(file_path): 
        with open(file_path, "r") as text_file: data_lines = text_file.readlines()
        all_numeric_lines = []
        if len(data_lines)==1: 
            data_lines_corrected = data_lines[0].split('Epoch')[1:]
            data_lines = ['Epoch'+e for e in data_lines_corrected]

        for data_line in data_lines:
            numerics = []
            for e in re.split(': | |\n|',data_line):
                try: 
                    float_e = float(e)
                    numerics.append(float_e)
                except: 
                    pass
            all_numeric_lines.append(numerics)
        np_all_numeric_lines = np.asarray(all_numeric_lines)
        list_of_np_numerics.append(np_all_numeric_lines)

# identifiers = ['WAE', 'WGAN-GP', 'PD-WGAN', 'Train. Data']
# colors = ['b', 'g', 'r', 'k', 'y']
# markers = ['d', 'v','h',  'o', 's']

identifiers = ['WGAN-GP', 'PD-WGAN', 'Train. Data']
colors = ['g', 'r', 'k', 'y']
markers = ['v', 'h',  'o', 's']

if mode == 'LeastEpochs':    
    least_max_epoch = 100000000000000
    for i, np_all_numeric_lines in enumerate(list_of_np_numerics):
        curr_max_epoch = np.max(np_all_numeric_lines[1:,0])
        if curr_max_epoch < least_max_epoch: least_max_epoch = curr_max_epoch

# plt.figure(figsize=(fig_width/my_dpi, fig_height/my_dpi), dpi=my_dpi)
plt.figure(figsize=(fig_width/(2*my_dpi), fig_height/my_dpi), dpi=my_dpi)
plt.cla()
y_label = 'Frechet Inception Distance (FID)'
x_label = 'Training Epochs'
min_y_val = 100000000000000
max_y_val = -100000000000000
for i, np_all_numeric_lines in enumerate(list_of_np_numerics):
    if mode == 'LeastEpochs':
        mask = np_all_numeric_lines[1:,0]<=least_max_epoch
        x_vals = np_all_numeric_lines[1:,0][mask]
        y_vals = np_all_numeric_lines[1:,1][mask]
    elif mode == 'EpochInterval':        
        mask_upper = np_all_numeric_lines[1:,0]<=max_epoch
        mask_lower = np_all_numeric_lines[1:,0]>=min_epoch        
        mask = mask_upper*mask_lower
        x_vals = np_all_numeric_lines[1:,0][mask]
        y_vals = np_all_numeric_lines[1:,1][mask]
    else:
        x_vals = np_all_numeric_lines[1:,0]
        y_vals = np_all_numeric_lines[1:,1]
    if np.min(y_vals)<min_y_val: min_y_val = np.min(y_vals)
    if np.max(y_vals)>max_y_val: max_y_val = np.max(y_vals)
    plt.plot(x_vals, y_vals, linewidth=2, linestyle='-', color=colors[i], label=identifiers[i], marker=markers[i], markersize=10)

y_range = (max_y_val-min_y_val)
plt.ylabel(y_label, fontsize=16)
plt.xlabel(x_label, fontsize=16)
plt.grid()
plt.legend(frameon=True)
plt.ylim((min_y_val-0.1*y_range, max_y_val+0.1*y_range ))
plt.xlim((0,1000))
plt.savefig(exp_folders[save_index]+'Visualization/fid_comparison.png', bbox_inches='tight', format='png', dpi=my_dpi, transparent=False)
print('Saving to path: ', exp_folders[save_index]+'Visualization/fid_comparison.png')


# plt.figure(figsize=(fig_width/my_dpi, fig_height/my_dpi), dpi=my_dpi)
plt.figure(figsize=(fig_width/(2*my_dpi), fig_height/my_dpi), dpi=my_dpi)
plt.cla()
y_label = 'Inception Score (IS)'
x_label = 'Training Epochs'
min_y_val = 100000000000000
max_y_val = -100000000000000
for i, np_all_numeric_lines in enumerate(list_of_np_numerics):
    if mode == 'LeastEpochs':
        mask = np_all_numeric_lines[1:,0]<=least_max_epoch
        x_vals = np_all_numeric_lines[1:,0][mask]
        y_vals = np_all_numeric_lines[1:,2][mask]
    elif mode == 'EpochInterval':        
        mask_upper = np_all_numeric_lines[1:,0]<=max_epoch
        mask_lower = np_all_numeric_lines[1:,0]>=min_epoch        
        mask = mask_upper*mask_lower
        x_vals = np_all_numeric_lines[1:,0][mask]
        y_vals = np_all_numeric_lines[1:,2][mask]
    else:
        x_vals = np_all_numeric_lines[1:,0]
        y_vals = np_all_numeric_lines[1:,2]
    if np.min(y_vals)<min_y_val: min_y_val = np.min(y_vals)
    if np.max(y_vals)>max_y_val: max_y_val = np.max(y_vals)
    plt.plot(x_vals, y_vals, linewidth=2, linestyle='-', color=colors[i], label=identifiers[i], marker=markers[i], markersize=10)

# i=3
# np_all_numeric_lines=list_of_np_numerics[0]
# if mode == 'LeastEpochs':
#     mask = np_all_numeric_lines[1:,0]<=least_max_epoch
#     x_vals = np_all_numeric_lines[1:,0][mask]
#     y_vals = np_all_numeric_lines[1:,4][mask]
# else:
#     x_vals = np_all_numeric_lines[1:,0]
#     y_vals = np_all_numeric_lines[1:,4]
# if np.min(y_vals)<min_y_val: min_y_val = np.min(y_vals)
# if np.max(y_vals)>max_y_val: max_y_val = np.max(y_vals)
# plt.plot(x_vals, y_vals, linewidth=2, linestyle='-', color=colors[i], label=identifiers[i], marker=markers[i], markersize=10)

y_range = (max_y_val-min_y_val)
plt.ylabel(y_label, fontsize=16)
plt.xlabel(x_label, fontsize=16)
plt.grid()
plt.legend(frameon=True)
plt.ylim((min_y_val-0.1*y_range, max_y_val+0.4*y_range ))
plt.xlim((0,1000))
plt.savefig(exp_folders[save_index]+'Visualization/is_comparison.png', bbox_inches='tight', format='png', dpi=my_dpi, transparent=False)
print('Saving to path: ', exp_folders[save_index]+'Visualization/is_comparison.png')

plt.close('all')



