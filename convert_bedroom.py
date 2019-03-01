import glob
import scipy
import scipy.misc
from PIL import Image
import pdb 
import os
from shutil import copyfile
import cv2

def read_cropped(filename, size, crop_size, interp):
	rgb = scipy.misc.imread(filename)
	x_start = int((rgb.shape[0]-crop_size)/2)
	y_start = int((rgb.shape[1]-crop_size)/2)
	rgb_cropped = rgb[x_start:x_start+crop_size,y_start:y_start+crop_size,:]
	rgb_scaled = scipy.misc.imresize(rgb_cropped, (size, size), interp=interp)
	return rgb_scaled

crop_size = 256

image_size = 64
# image_size = 128

# interp='nearest'
# interp='bilinear'
# interp='bicubic'
interp='cubic'
# interp='lanczos'

source_path = '/home/mcgemici/lsun_bedroom_data/train_webp/'
dest_path = '/home/mcgemici/lsun_bedroom_data/train_'+str(image_size)+'_'+interp+'/'
file_path = source_path+'*.webp'
files = glob.glob(file_path)

if not os.path.exists(dest_path): os.makedirs(dest_path)
for i, f in enumerate(files):
	target_path = dest_path+f[len(source_path):]
	print(i, len(files), target_path)
	cropped = read_cropped(f, size=image_size, crop_size=crop_size, interp=interp)
	scipy.misc.imsave(target_path, cropped)



