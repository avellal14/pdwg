import glob
import scipy
import scipy.misc
import pdb 
import os

def read_cropped(filename, size, crop_size, interp):
	rgb = scipy.misc.imread(filename)
	x_start = int((rgb.shape[0]-crop_size)/2)
	y_start = int((rgb.shape[1]-crop_size)/2)
	rgb_cropped = rgb[x_start:x_start+crop_size,y_start:y_start+crop_size,:]
	rgb_scaled = scipy.misc.imresize(rgb_cropped, (size, size), interp=interp)
	return rgb_scaled

image_size=64
crop_size = 500
# interp='bilinear'
interp='nearest'
source_path = '/home/mcgemici/flowers_data/jpg/'
dest_path = '/home/mcgemici/flowers_data/flowers_'+str(image_size)+'_'+interp+'/class1/'
file_path = source_path+'*.jpg'
files = glob.glob(file_path)
for i, f in enumerate(files):
	print(i, len(files))
	if not os.path.exists(dest_path): os.makedirs(dest_path)
	cropped = read_cropped(f, size=image_size, crop_size=crop_size, interp=interp)
	scipy.misc.imsave(dest_path+f[len(source_path):], cropped)
