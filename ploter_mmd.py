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
from pathlib import Path
import csv

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

file_path = str(Path.home())+'/ExperimentalResults/MMD_test/results.txt'

# save_index = -1
# # mode = 'LeastEpochs'
# # mode = 'Full'
# mode = 'EpochInterval'
# min_epoch = 200
# max_epoch = 1500
# list_of_np_numerics = []

# if os.path.exists(file_path): 
#     with open(file_path, "r") as text_file: data_lines = text_file.readlines()
#     all_numeric_lines = []
#     if len(data_lines)==1: 
#         data_lines_corrected = data_lines[0].split('Epoch')[1:]
#         data_lines = ['Epoch'+e for e in data_lines_corrected]

#     for data_line in data_lines:
#         numerics = []
#         for e in re.split(': | |\n|',data_line):
#             try: 
#                 float_e = float(e)
#                 numerics.append(float_e)
#             except: 
#                 pass
#         all_numeric_lines.append(numerics)
#     np_all_numeric_lines = np.asarray(all_numeric_lines)
#     list_of_np_numerics.append(np_all_numeric_lines)

all_data = {'i': [], 'MMD': [], 'Rec': [], 'Overall': [], 'Lambda': []}
with open(file_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        all_data['i'].append(float(row[0]))
        all_data['Overall'].append(float(row[1]))
        all_data['MMD'].append(float(row[2]))
        all_data['Rec'].append(float(row[3]))
        all_data['Lambda'].append(float(row[4]))

# identifiers = ['Lambda', 'Rec', 'MMD', 'Overall']
identifiers = ['Lambda', 'Rec', 'MMD']
colors = ['g', 'r', 'b', 'k']
markers = ['v', 'h',  'o', 's']

# plt.figure(figsize=(fig_width/my_dpi, fig_height/my_dpi), dpi=my_dpi)
plt.figure(figsize=(fig_width/(2*my_dpi), fig_height/my_dpi), dpi=my_dpi)
plt.cla()
y_label = 'MMD Test'
x_label = 'Training Iterations'
# min_y_val = 100000000000000
# max_y_val = -100000000000000
min_y_val = 100000000000000
max_y_val = -100000000000000
for i, identifier in enumerate(identifiers):
    x_vals = all_data['i']
    y_vals = all_data[identifier]
    if min(y_vals)<min_y_val: min_y_val = min(y_vals)
    if max(y_vals)>max_y_val: max_y_val = max(y_vals)
    plt.plot(x_vals, y_vals, linewidth=2, linestyle='-', color=colors[i], label=identifiers[i], marker=markers[i], markersize=10)

y_range = (max_y_val-min_y_val)
plt.ylabel(y_label, fontsize=16)
plt.xlabel(x_label, fontsize=16)
plt.grid()
plt.legend(frameon=True)
# plt.ylim((min_y_val-0.1*y_range, max_y_val+0.1*y_range ))
plt.ylim((min_y_val-0.1*y_range, min(5, max_y_val+0.1*y_range) ))
plt.xlim((0,3000))
plt.savefig(str(Path.home())+'/ExperimentalResults/MMD_test/results.png', bbox_inches='tight', format='png', dpi=my_dpi, transparent=False)
print('Saving to path: ', str(Path.home())+'/ExperimentalResults/MMD_test/results.png')

# # plt.figure(figsize=(fig_width/my_dpi, fig_height/my_dpi), dpi=my_dpi)
# plt.figure(figsize=(fig_width/(2*my_dpi), fig_height/my_dpi), dpi=my_dpi)
# plt.cla()
# y_label = 'Inception Score (IS)'
# x_label = 'Training Epochs'
# min_y_val = 100000000000000
# max_y_val = -100000000000000
# for i, np_all_numeric_lines in enumerate(list_of_np_numerics):
#     if mode == 'LeastEpochs':
#         mask = np_all_numeric_lines[1:,0]<=least_max_epoch
#         x_vals = np_all_numeric_lines[1:,0][mask]
#         y_vals = np_all_numeric_lines[1:,2][mask]
#     elif mode == 'EpochInterval':        
#         mask_upper = np_all_numeric_lines[1:,0]<=max_epoch
#         mask_lower = np_all_numeric_lines[1:,0]>=min_epoch        
#         mask = mask_upper*mask_lower
#         x_vals = np_all_numeric_lines[1:,0][mask]
#         y_vals = np_all_numeric_lines[1:,2][mask]
#     else:
#         x_vals = np_all_numeric_lines[1:,0]
#         y_vals = np_all_numeric_lines[1:,2]
#     if np.min(y_vals)<min_y_val: min_y_val = np.min(y_vals)
#     if np.max(y_vals)>max_y_val: max_y_val = np.max(y_vals)
#     plt.plot(x_vals, y_vals, linewidth=2, linestyle='-', color=colors[i], label=identifiers[i], marker=markers[i], markersize=10)

# # i=3
# # np_all_numeric_lines=list_of_np_numerics[0]
# # if mode == 'LeastEpochs':
# #     mask = np_all_numeric_lines[1:,0]<=least_max_epoch
# #     x_vals = np_all_numeric_lines[1:,0][mask]
# #     y_vals = np_all_numeric_lines[1:,4][mask]
# # else:
# #     x_vals = np_all_numeric_lines[1:,0]
# #     y_vals = np_all_numeric_lines[1:,4]
# # if np.min(y_vals)<min_y_val: min_y_val = np.min(y_vals)
# # if np.max(y_vals)>max_y_val: max_y_val = np.max(y_vals)
# # plt.plot(x_vals, y_vals, linewidth=2, linestyle='-', color=colors[i], label=identifiers[i], marker=markers[i], markersize=10)

# y_range = (max_y_val-min_y_val)
# plt.ylabel(y_label, fontsize=16)
# plt.xlabel(x_label, fontsize=16)
# plt.grid()
# plt.legend(frameon=True)
# plt.ylim((min_y_val-0.1*y_range, max_y_val+0.4*y_range ))
# plt.xlim((0,1000))
# plt.savefig(exp_folders[save_index]+'Visualization/is_comparison.png', bbox_inches='tight', format='png', dpi=my_dpi, transparent=False)
# print('Saving to path: ', exp_folders[save_index]+'Visualization/is_comparison.png')

# plt.close('all')



