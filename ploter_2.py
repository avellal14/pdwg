
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

exp_folder = './experimentsLast-PDWGANCannon-CIFAR10/8de59b8d2c3a4ec7976bf4f689fb2477/'
exp_folder_2 = './OlderExperiments/fifthSet/experiments128-WAEVanilla-CIFAR10/107c24e9367f490fa5ffa2f2a49efb41/' # WAE
train_data_all = None
test_data_all = None
print('Extracting from: ', exp_folder)
image_file_name = 'ot_dist'

train_trace_file_path = exp_folder+'/training_primal_dual.txt'
test_trace_file_path = exp_folder+'/test_primal_dual.txt'    
WAE_train_trace_file_path = exp_folder_2+'/training_primal_dual.txt'

if os.path.exists(WAE_train_trace_file_path): 
    with open(WAE_train_trace_file_path, "r") as text_file:
        training_data_lines = text_file.readlines()
    lines_split = []
    for line in training_data_lines: lines_split.append(line.split()[1::2])

    # WAE_train_data = np.asarray([line.strip().split(',') for line in training_data_lines])
    WAE_train_data = np.asarray(lines_split)
    WAE_train_primal = WAE_train_data[:,0].astype(float)
    WAE_train_dual = WAE_train_data[:,1].astype(float)

if os.path.exists(train_trace_file_path): 
    with open(train_trace_file_path, "r") as text_file:
        training_data_lines = text_file.readlines()
    lines_split = []
    for line in training_data_lines: lines_split.append(line.split()[1::2])

    # train_data = np.asarray([line.strip().split(',') for line in training_data_lines])
    train_data = np.asarray(lines_split)
    train_primal = train_data[:,0].astype(float)
    train_dual = train_data[:,1].astype(float)
    train_primal_dual = WAE_train_primal#0.5*(train_primal+train_dual)

    identifiers = ['PD-WGAN [Primal]', 'PD-WGAN [Dual]', 'WAE [Primal]']
    colors = ['r', 'g', 'b', 'k', 'y']
    y_label = 'Optimal Transport Distance (Approximate)'
    x_label = 'Training Steps'
    plt.cla()
    plt.plot(train_primal, linewidth=2, linestyle='-', color=colors[0], label=identifiers[0])
    plt.plot(train_dual, linewidth=2, linestyle='-', color=colors[1], label=identifiers[1])
    plt.plot(train_primal_dual, linewidth=2, linestyle='-', color=colors[2], label=identifiers[2])
    plt.ylabel(y_label, fontsize=16)
    plt.xlabel(x_label, fontsize=16)
    plt.grid()
    plt.legend(frameon=True)
    plt.ylim((0,15))
    plt.xlim((0,200))
    plt.savefig(exp_folder+'/train_'+image_file_name+'.png', bbox_inches='tight', format='png', dpi=400, transparent=False)

## Regularizers

exp_folder = './experimentsLast-PDWGANCannon-CIFAR10/54103a5357ec46eaaea3c3d7125c4b8a/' # PDWGAN
exp_folder_2 = './experimentsLast-WGANGPCannon-CIFAR10/3f5b283edea7462eb2151a255adc69cc/' # WGAN-GP
train_data_all = None
test_data_all = None
print('Extracting from: ', exp_folder)
image_file_name_1 = 'coupling_reg'
image_file_name_2 = 'trivial_reg'

PDWGAN_train_trace_file_path = exp_folder+'/training_primal_dual.txt'
WGANGP_train_trace_file_path = exp_folder_2+'/training_primal_dual.txt'

if os.path.exists(PDWGAN_train_trace_file_path): 
    with open(PDWGAN_train_trace_file_path, "r") as text_file:
        training_data_lines = text_file.readlines()
    lines_split = []
    for line in training_data_lines: lines_split.append(line.split()[1::2])

    # WAE_train_data = np.asarray([line.strip().split(',') for line in training_data_lines])
    PDWGAN_train_data = np.asarray(lines_split)
    PDWGAN_train_coupling = PDWGAN_train_data[:,2].astype(float)
    PDWGAN_train_trivial = PDWGAN_train_data[:,3].astype(float)


if os.path.exists(WGANGP_train_trace_file_path): 
    with open(WGANGP_train_trace_file_path, "r") as text_file:
        training_data_lines = text_file.readlines()
    lines_split = []
    for line in training_data_lines: lines_split.append(line.split()[1::2])

    # WAE_train_data = np.asarray([line.strip().split(',') for line in training_data_lines])
    WGANGP_train_data = np.asarray(lines_split)
    WGANGP_train_coupling = WGANGP_train_data[:,2].astype(float)
    WGANGP_train_trivial = WGANGP_train_data[:,3].astype(float)


if os.path.exists(PDWGAN_train_trace_file_path): 
    identifiers = ['PD-WGAN', 'WGAN-GP']
    colors = ['r', 'g']
    y_label = 'Optimal Coupling Gradient Vector Penalty'
    x_label = 'Training Steps'
    
    plt.figure(figsize=(5,8))
    plt.cla()
    plt.plot(PDWGAN_train_coupling, linewidth=2, linestyle='-', color=colors[0], label=identifiers[0])
    plt.plot(WGANGP_train_coupling, linewidth=2, linestyle='-', color=colors[1], label=identifiers[1])
    plt.ylabel(y_label, fontsize=16)
    plt.xlabel(x_label, fontsize=16)
    plt.grid()
    plt.legend(frameon=True, loc='center right')
    plt.ylim((-0.04,1.5))
    plt.xlim((0,150))
    plt.savefig(exp_folder+'/train_'+image_file_name_1+'_2.png', bbox_inches='tight', format='png', dpi=400, transparent=False)

    identifiers = ['PD-WGAN', 'WGAN-GP']
    colors = ['r', 'g']
    y_label = 'Trivial Coupling Gradient Norm Penalty'
    x_label = 'Training Steps'

    plt.figure(figsize=(5,8))
    plt.cla()
    plt.plot(PDWGAN_train_trivial, linewidth=2, linestyle='-', color=colors[0], label=identifiers[0])
    plt.plot(WGANGP_train_trivial, linewidth=2, linestyle='-', color=colors[1], label=identifiers[1])
    plt.ylabel(y_label, fontsize=16)
    plt.xlabel(x_label, fontsize=16)
    plt.grid()
    plt.legend(frameon=True, loc='center right')
    plt.ylim((-0.04,0.85))
    plt.xlim((0,150))
    plt.savefig(exp_folder+'/train_'+image_file_name_2+'_2.png', bbox_inches='tight', format='png', dpi=400, transparent=False)





# if os.path.exists(test_trace_file_path): 
#     with open(test_trace_file_path, "r") as text_file:
#         test_data_lines =text_file.readlines()
#     test_data = np.asarray([line.strip().split(',') for line in test_data_lines])
#     test_primal = test_data[:,0].astype(float)
#     test_dual = test_data[:,1].astype(float)
#     test_primal_dual = 0.5*(test_primal+test_dual)

#     plt.cla()
#     plt.plot(test_primal, linewidth=2, linestyle='-', color=colors[0], label=identifiers[0])
#     plt.plot(test_dual, linewidth=2, linestyle='-', color=colors[1], label=identifiers[1])
#     plt.plot(test_primal_dual, linewidth=2, linestyle=':', color=colors[2], label=identifiers[2])
#     plt.ylabel(y_label, fontsize=16)
#     plt.grid()
#     plt.legend()
#     # plt.ylim((-1000,200))
#     plt.xlim((0,800))
#     plt.savefig(exp_folder+'/test_'+image_file_name+'.png', bbox_inches='tight', format='png', dpi=400, transparent=False)



# #######################################################       CREATE GIFS       ########################################################
# max_gif_steps = 200
# delay = 20
# for i, exp_folder in enumerate(onlyfolders):
#     print('\nCreating gifs for experiment: ', exp_folder)
#     result_paths = ''
#     for subdir in ['dataset_plotter_data_only', 'dataset_plotter_data_real', 'plot2D_dist', 'plot2D_dist_b_labeled', 'barplot', 'Train/Fixed/0']:
#         subdir_path = all_experiments_dir+exp_folder+'/Visualization/'+subdir+'/*.png'
#         files = glob.glob(subdir_path)
#         try: order = list(np.argsort([int(filename.split('_')[-3]) for filename in files]))
#         except: order = list(np.argsort([int(filename.split('_')[-1][:-len('.png')]) for filename in files]))
#         ordered_files = [files[ind] for ind in order]
#         ordered_files = ordered_files[:max_gif_steps]
#         ordered_files_str = ''
#         for f in ordered_files: ordered_files_str = ordered_files_str + ' ' + f
        
#         print('Creating gif for', subdir, '(Number of images ==> ', len(ordered_files))
#         # os.system('convert -resize 800x800 -delay '+str(delay)+' -loop 0 '+ordered_files_str+' '+all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif')
#         # os.system('convert -resize 205x800 -delay '+str(delay)+' -loop 0 '+ordered_files_str+' '+all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif')
#         # os.system('convert --compress None -quality 100 -delay '+str(delay)+' -loop 0 '+ordered_files_str+' '+all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif')
#         os.system('convert -quality 100 -delay '+str(delay)+' -loop 0 '+ordered_files_str+' '+all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif')
#         result_paths = result_paths + ' ' +all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif'
    
#     print('Generating aggregate gif.')    
#     aggregate_path = all_experiments_dir+exp_folder+'/Visualization/dataset_plotter_data_only.gif'
#     # for i, subdir in enumerate(['dataset_plotter_data_real', 'plot2D_dist', 'plot2D_dist_b_labeled']):
#     for i, subdir in enumerate(['dataset_plotter_data_real']):
#         bag_left_path = aggregate_path
#         bag_right_path = all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif'
#         aggregate_path = all_experiments_dir+exp_folder+'/Visualization/'+exp_folder+'_'+str(i)+'.gif'
#         os.system("convert "+bag_left_path+"'[0]' -coalesce \( "+bag_right_path+"'[0]' -coalesce \) +append -channel A -evaluate set 0 +channel "+bag_left_path+" -coalesce -delete 0 null: \( "+bag_right_path+" -coalesce \) -gravity East  -layers Composite    "+aggregate_path)

# print('\nGenerating aggregate gif for all experiments.')  
# aggregate_path = all_experiments_dir+onlyfolders[0]+'/Visualization/'+onlyfolders[0]+'_0.gif'
# for i, exp_folder in enumerate(onlyfolders[1:]):
#     bag_left_path = aggregate_path
#     bag_right_path = all_experiments_dir+exp_folder+'/Visualization/'+exp_folder+'_0.gif'
#     aggregate_path = all_experiments_dir+'all_experiments_'+str(i)+'.gif'
#     os.system("convert "+bag_left_path+"'[0]' -coalesce \( "+bag_right_path+"'[0]' -coalesce \) -append -channel A -evaluate set 0 +channel "+bag_left_path+" -coalesce -delete 0 null: \( "+bag_right_path+" -coalesce \) -gravity South  -layers Composite    "+aggregate_path)












































# for col in range(train_data.shape[1]):
#     plt.plot(train_data[:, col], linewidth=2, color='g', linestyle='-')
#     plt.grid()
#     plt.savefig(all_experiments_dir+'training_traces_'+str(col)+'.png', bbox_inches='tight', format='png', dpi=400, transparent=False)

# plt.plot(test_data, linewidth=2, color='g', linestyle='-')
# # for col in range(test_data.shape[1]):
# #     plt.plot(test_data[:, col], linewidth=2, color='g', linestyle='-')
# #     plt.grid()
# plt.savefig(all_experiments_dir+'test_traces_.png', bbox_inches='tight', format='png', dpi=400, transparent=False)

# for exp_folder in onlyfolders:
#     train_trace_file_path = all_experiments_dir+exp_folder+'/training_traces.txt'
#     test_trace_file_path = all_experiments_dir+exp_folder+'/test_traces.txt'
#     with open(train_trace_file_path, "r") as text_file:
#         training_data_lines =text_file.readlines()
#     train_data = np.asarray([line.strip().split(',') for line in training_data_lines])
#     for col in range(train_data.shape[1]):
#         plt.cla()
#         plt.plot(train_data[:, col], linewidth=2, color='g', linestyle='-')
#         plt.grid()
#         plt.savefig(all_experiments_dir+'training_traces_'+str(col)+'.png', bbox_inches='tight', format='png', dpi=400, transparent=False)
    
#     with open(test_trace_file_path, "r") as text_file:
#         test_data_lines =text_file.readlines()
#     test_data = np.asarray([line.strip().split(',') for line in test_data_lines])
#     for col in range(test_data.shape[1]):
#         plt.cla()
#         plt.plot(test_data[:, col], linewidth=2, color='g', linestyle='-')
#         plt.grid()
#         plt.savefig(all_experiments_dir+'test_traces_'+str(col)+'.png', bbox_inches='tight', format='png', dpi=400, transparent=False)
# pdb.set_trace()
