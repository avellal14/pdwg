
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
from pathlib import Path

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


max_gif_steps = 300
delay = 20
duplicate_first_frame = 10
duplicate_last_frame = 10
gt_augment = True

main_folder = str(Path.home())+'/sgld_vis/'
if os.path.exists(main_folder+'gt/gt_temp.png'): os.remove(main_folder+'gt/gt_temp.png')

for subdir in ['im']:
# for subdir in ['all', 'gt',  'im']:
	exp_folder = main_folder+ subdir +'/'
	result_folder = main_folder+ subdir +'/'
	print('\nProcessing folder: ', main_folder+ subdir +'/')

	files = glob.glob(main_folder+ subdir +'/*.png')

	try: order = list(np.argsort([int(filename.split('_')[-3]) for filename in files]))
	except: order = list(np.argsort([int(filename.split('_')[-1][:-len('.png')]) for filename in files]))
	ordered_files = [files[ind] for ind in order]
	ordered_files = ordered_files[:max_gif_steps]
	ordered_files_str = ''
		
	if duplicate_first_frame > 0: 
		att = [ordered_files[0],]*duplicate_first_frame
		ordered_files = att+(ordered_files[1:])
	if duplicate_last_frame > 0: 
		att = [ordered_files[-1],]*duplicate_last_frame
		ordered_files = (ordered_files[:-1])+att

	for f in ordered_files: ordered_files_str = ordered_files_str + ' ' + f

	last_file_path = ordered_files[-1]
	last_image = plt.imread(last_file_path)
	
	if subdir == 'all':
		n_steps = 20
		last_image_in = last_image[:, :last_image.shape[0],:]
		last_image_out = last_image[:, last_image.shape[0]:2*last_image.shape[0],:]
		last_image_target = last_image[:, 2*last_image.shape[0]:2*last_image.shape[0]+last_image.shape[0],:]
		mask_1 = (np.concatenate([np.linspace(0,1,n_steps), np.linspace(1,0,n_steps)]))[:, np.newaxis, np.newaxis, np.newaxis]
		mask_2 = 1-mask_1

		if not os.path.exists(exp_folder+'in_target_interp/'): os.makedirs(exp_folder+'in_target_interp/')
		in_target_interp_ims = ((mask_1*last_image_in[np.newaxis, :, :, :])+(mask_2*last_image_target[np.newaxis, :, :, :]))
		in_target_interp_files_str = ''
		for i in range(in_target_interp_ims.shape[0]):
			file_path = exp_folder+'in_target_interp/in_target_interp_'+str(i+1)+'.png'
			plt.imsave(file_path, in_target_interp_ims[i, ...])
			in_target_interp_files_str = in_target_interp_files_str + ' ' + file_path

		print('Creating gif for', exp_folder+'in_target_interp/', '(Number of images ==> ', str(in_target_interp_ims.shape[0]))
		os.system('convert -quality 100 -delay '+str(delay)+' -loop 0 '+in_target_interp_files_str+' '+exp_folder+'in_target_interp/in_target_interp.gif')

		if not os.path.exists(exp_folder+'out_target_interp/'): os.makedirs(exp_folder+'out_target_interp/')
		out_target_interp_ims = ((mask_1*last_image_out[np.newaxis, :, :, :])+(mask_2*last_image_target[np.newaxis, :, :, :]))
		out_target_interp_files_str = ''
		for i in range(out_target_interp_ims.shape[0]):
			file_path = exp_folder+'out_target_interp/out_target_interp_'+str(i+1)+'.png'
			plt.imsave(file_path, out_target_interp_ims[i, ...])
			out_target_interp_files_str = out_target_interp_files_str + ' ' + file_path

		print('Creating gif for', exp_folder+'out_target_interp/', '(Number of images ==> ', str(out_target_interp_ims.shape[0]))
		os.system('convert -quality 100 -delay '+str(delay)+' -loop 0 '+out_target_interp_files_str+' '+exp_folder+'out_target_interp/out_target_interp.gif')

	print('Creating gif for', exp_folder, '(Number of images ==> ', len(ordered_files))
	# os.system('convert -resize 800x800 -delay '+str(delay)+' -loop 0 '+ordered_files_str+' '+all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif')
	# os.system('convert -resize 205x800 -delay '+str(delay)+' -loop 0 '+ordered_files_str+' '+all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif')
	# os.system('convert --compress None -quality 100 -delay '+str(delay)+' -loop 0 '+ordered_files_str+' '+all_experiments_dir+exp_folder+'/Visualization/'+subdir+'.gif')
	os.system('convert -quality 100 -delay '+str(delay)+' -loop 0 '+ordered_files_str+' '+result_folder+'all.gif')


















