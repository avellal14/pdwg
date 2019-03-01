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

# path_sources = ['./experiments*/*/', './EEEexperiments*/*/',]
path_sources = ['/Users/MeVlana/CloudExperiments/experiments*/*/', '/Users/MeVlana/CloudExperiments/EEEexperiments*/*/',]

for path_source in path_sources:
    list_experiment_folders = glob.glob(path_source)
    for exp_folder in list_experiment_folders:
        print(exp_folder)
        spec_file_path = exp_folder+'Specs.txt'
        target_file_path = exp_folder+'Listed_Specs.txt'
        with open(spec_file_path, "r") as text_file: data_lines = text_file.readlines()
        all_data_str = ''.join(data_lines)
        all_data_str = all_data_str.split('Namespace', 1)[-1]
        all_data_str = all_data_str.rstrip("\n")
        all_data_str = all_data_str[1:-1]
        
        split_list = all_data_str.split(',')
        full_list = []
        curr = []
        for e in split_list:
            if '=' in e:
                full_list.append(''.join(curr))
                curr = []
            curr.append(e)
        full_list = full_list[1:]
        pro_full_list = []
        for e in full_list: 
            pro_full_list.append(e.strip().replace("'", '"')) 
        pro_full_list.sort()
        with open(target_file_path, "w") as text_file: text_file.write('\n'.join(pro_full_list))















