
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


all_experiments_dir = './PlotGroups/group18/'
if not os.path.exists(all_experiments_dir): os.makedirs(all_experiments_dir)
onlyfolders = [f for f in listdir(all_experiments_dir) if not isfile(join(all_experiments_dir, f))]

# properties_to_seek = ['transformedQ', 'learning_rate']
# properties = []
# train_data_all = None
# test_data_all = None
# for i, exp_folder in enumerate(onlyfolders):
#     print('Extracting from: ', exp_folder)
#     specs_file_path = all_experiments_dir+exp_folder+'/Specs.txt'
#     with open(specs_file_path, "r") as text_file:
#         specs_file_lines =text_file.readlines()

#     specs_file_lines = [e.strip() for e in specs_file_lines]
#     curr_property = {}
#     for e in (''.join(specs_file_lines)).split(','):
#         for property_to_seek in properties_to_seek:
#             if property_to_seek in e:
#                 # curr_property[property_to_seek] = e.split('=')[1]
#                 curr_property[property_to_seek] = e
#     properties.append(curr_property)

#     train_trace_file_path = all_experiments_dir+exp_folder+'/training_traces.txt'
#     with open(train_trace_file_path, "r") as text_file:
#         training_data_lines =text_file.readlines()
#     train_data = np.asarray([line.strip().split(',') for line in training_data_lines])

#     if train_data_all is None:
#         train_data_all = np.empty((train_data.shape[0], len(onlyfolders), train_data.shape[1]))
#         train_data_all.fill(np.NAN)
#         train_data_all[:, i, :] = train_data
#     elif train_data_all.shape[0]<train_data.shape[0]:
#         train_data_all_temp = np.empty((train_data.shape[0], len(onlyfolders), train_data.shape[1]))
#         train_data_all_temp.fill(np.NAN)
#         train_data_all_temp[:train_data_all.shape[0], :i, :] = train_data_all[:, :i, :]
#         train_data_all_temp[:, i, :] = train_data 
#         train_data_all = train_data_all_temp
#     else: 
#         train_data_all[:train_data.shape[0],i,:] = train_data

#     test_trace_file_path = all_experiments_dir+exp_folder+'/test_traces.txt'    
#     with open(test_trace_file_path, "r") as text_file:
#         test_data_lines =text_file.readlines()
#     test_data = np.asarray([line.strip().split(',') for line in test_data_lines])
    
#     if test_data_all is None:
#         test_data_all = np.empty((test_data.shape[0], len(onlyfolders), test_data.shape[1]))
#         test_data_all.fill(np.NAN)
#         test_data_all[:, i, :] = test_data
#     elif test_data_all.shape[0]<test_data.shape[0]:
#         test_data_all_temp = np.empty((test_data.shape[0], len(onlyfolders), test_data.shape[1]))
#         test_data_all_temp.fill(np.NAN)
#         test_data_all_temp[:test_data_all.shape[0], :i, :] = test_data_all[:, :i, :]
#         test_data_all_temp[:, i, :] = test_data 
#         test_data_all = test_data_all_temp
#     else: 
#         test_data_all[:test_data.shape[0],i,:] = test_data

# identifiers = [', '.join(list(e.values())) for e in properties]
# colors = ['r', 'g', 'b', 'k', 'y']
# col_names = ['elbo', 'likelihood', 'kl', 'temp']
# for col in range(train_data_all.shape[2]):
#     plt.cla()
#     for i in range(train_data_all.shape[1]):
#         plt.plot(train_data_all[:, i, col], linewidth=2, linestyle='-', color=colors[i], label=identifiers[i])
#     plt.ylabel(col_names[col], fontsize=16)
#     plt.grid()
#     plt.legend()
#     plt.ylim((-1000,200))
#     plt.xlim((0,1500))
#     plt.savefig(all_experiments_dir+'train_'+col_names[col]+'.png', bbox_inches='tight', format='png', dpi=400, transparent=False)

# for col in range(test_data_all.shape[2]):
#     plt.cla()
#     for i in range(train_data_all.shape[1]):
#         plt.plot(test_data_all[:, i, col], linewidth=2, linestyle='-', color=colors[i], label=identifiers[i])
#     plt.ylabel(col_names[col], fontsize=16)
#     plt.grid()
#     plt.legend()
#     plt.ylim((-1000,200))
#     plt.xlim((0,70))
#     plt.savefig(all_experiments_dir+'test_'+col_names[col]+'.png', bbox_inches='tight', format='png', dpi=400, transparent=False)



#######################################################       CREATE GIFS       ########################################################
max_gif_steps = 200
delay = 20
for i, exp_folder in enumerate(onlyfolders):
    print('\nCreating gifs for experiment: ', exp_folder)
    result_paths = ''
    for subdir in ['dataset_plotter_data_only', 'dataset_plotter_data_real', 'plot2D_dist', 'plot2D_dist_b_labeled', 'barplot', 'Train/Fixed/0']:
        subdir_path = all_experiments_dir+exp_folder+'/Visualization/'+subdir+'/*.png'
        files = glob.glob(subdir_path)
        try: order = list(np.argsort([int(filename.split('_')[-3]) for filename in files]))
        except: order = list(np.argsort([int(filename.split('_')[-1][:-len('.png')]) for filename in files]))
        ordered_files = [files[ind] for ind in order]
        ordered_files = ordered_files[:max_gif_steps]
        ordered_files_str = ''
        for f in ordered_files: ordered_files_str = ordered_files_str + ' ' + f
        
        print('Creating gif for', subdir, '(Number of images ==> ', len(ordered_files))
        # os.system('convert -resize 800x800 -delay '+str(delay)+' -loop 0 '+ordered_files_str+' '+all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif')
        # os.system('convert -resize 205x800 -delay '+str(delay)+' -loop 0 '+ordered_files_str+' '+all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif')
        # os.system('convert --compress None -quality 100 -delay '+str(delay)+' -loop 0 '+ordered_files_str+' '+all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif')
        os.system('convert -quality 100 -delay '+str(delay)+' -loop 0 '+ordered_files_str+' '+all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif')
        result_paths = result_paths + ' ' +all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif'
    
    print('Generating aggregate gif.')    
    aggregate_path = all_experiments_dir+exp_folder+'/Visualization/dataset_plotter_data_only.gif'
    # for i, subdir in enumerate(['dataset_plotter_data_real', 'plot2D_dist', 'plot2D_dist_b_labeled']):
    for i, subdir in enumerate(['dataset_plotter_data_real']):
        bag_left_path = aggregate_path
        bag_right_path = all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif'
        aggregate_path = all_experiments_dir+exp_folder+'/Visualization/'+exp_folder+'_'+str(i)+'.gif'
        os.system("convert "+bag_left_path+"'[0]' -coalesce \( "+bag_right_path+"'[0]' -coalesce \) +append -channel A -evaluate set 0 +channel "+bag_left_path+" -coalesce -delete 0 null: \( "+bag_right_path+" -coalesce \) -gravity East  -layers Composite    "+aggregate_path)

print('\nGenerating aggregate gif for all experiments.')  
aggregate_path = all_experiments_dir+onlyfolders[0]+'/Visualization/'+onlyfolders[0]+'_0.gif'
for i, exp_folder in enumerate(onlyfolders[1:]):
    bag_left_path = aggregate_path
    bag_right_path = all_experiments_dir+exp_folder+'/Visualization/'+exp_folder+'_0.gif'
    aggregate_path = all_experiments_dir+'all_experiments_'+str(i)+'.gif'
    os.system("convert "+bag_left_path+"'[0]' -coalesce \( "+bag_right_path+"'[0]' -coalesce \) -append -channel A -evaluate set 0 +channel "+bag_left_path+" -coalesce -delete 0 null: \( "+bag_right_path+" -coalesce \) -gravity South  -layers Composite    "+aggregate_path)












































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
