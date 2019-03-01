
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


# all_experiments_dir = './PlotGroups/group18/'
# onlyfolders = [f for f in listdir(all_experiments_dir) if not isfile(join(all_experiments_dir, f))]

#######################################################       CREATE GIFS       ########################################################

delay = 20
first_n_images = 200
default_max_n_images = 50
rate = 2

# exp_folders = [
# 			   './experimentsCELEB/b48230d128e9404287f7c99c01f931f6/Visualization/test_fixed_sample/',
# 			   './experimentsCELEB/b48230d128e9404287f7c99c01f931f6/Visualization/test_real_sample_interpolated/',
# 			   './experimentsCELEB/b48230d128e9404287f7c99c01f931f6/Visualization/test_real_sample_reconst/',
# 			   ]

exp_folders = [
			   './experimentsMNIST/univApproxWithoutspatial/GAN/Visualization/test_fixed_sample/',
			   './experimentsMNIST/univApproxWithoutspatial/MMD/Visualization/test_fixed_sample/',
			   './experimentsMNIST/univApproxWithoutspatial/NonsaturatingGAN/Visualization/test_fixed_sample/',
			   ]

for exp_folder in exp_folders:
	print('\nCreating gifs for experiment: ', exp_folder)
	paths = glob.glob(exp_folder+'*.png')
	filenames = [e[len(exp_folder):] for e in paths]
	numbers = [[int(s) for s in e.split('_') if s.isdigit()][0] for e in filenames]
	order = list(np.argsort(numbers))
	ordered_paths = [paths[ind] for ind in order]
	ordered_paths = ordered_paths[:first_n_images]
	if rate is None: rate = math.ceil(float(len(ordered_paths))/float(default_max_n_images))
	ordered_paths = ordered_paths[0::rate]

	ordered_paths_str = ''
	for f in ordered_paths: ordered_paths_str += ' ' + f

	print('Creating gif for', exp_folder, '(Number of images ==> ', len(ordered_paths))
	# os.system('convert -resize 800x800 -delay '+str(delay)+' -loop 0 '+ordered_files_str+' '+all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif')
	# os.system('convert -resize 205x800 -delay '+str(delay)+' -loop 0 '+ordered_files_str+' '+all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif')
	# os.system('convert --compress None -quality 100 -delay '+str(delay)+' -loop 0 '+ordered_files_str+' '+all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif')
	os.system('convert -quality 100 -delay '+str(delay)+' -loop 0 '+ordered_paths_str+' '+exp_folder+'aggregate.gif')




# print('Generating aggregate gif.')    
# aggregate_path = all_experiments_dir+exp_folder+'/Visualization/dataset_plotter_data_only.gif'
# # for i, subdir in enumerate(['dataset_plotter_data_real', 'plot2D_dist', 'plot2D_dist_b_labeled']):
# for i, subdir in enumerate(['dataset_plotter_data_real']):
#     bag_left_path = aggregate_path
#     bag_right_path = all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif'
#     aggregate_path = all_experiments_dir+exp_folder+'/Visualization/'+exp_folder+'_'+str(i)+'.gif'
#     os.system("convert "+bag_left_path+"'[0]' -coalesce \( "+bag_right_path+"'[0]' -coalesce \) +append -channel A -evaluate set 0 +channel "+bag_left_path+" -coalesce -delete 0 null: \( "+bag_right_path+" -coalesce \) -gravity East  -layers Composite    "+aggregate_path)

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
