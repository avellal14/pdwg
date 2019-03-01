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

image_size = 128
crop_size = 150
# interp='nearest'
# interp='bilinear'
# interp='bicubic'
interp='cubic'
# interp='lanczos'

source_path = '/home/mcgemici/lsun_celebA_data/celebA2/splits/'
dest_path = '/home/mcgemici/lsun_celebA_data/celebA_'+str(image_size)+'_'+interp+'/splits/'
for typ in ['train/', 'test/', 'valid/']:
	file_path = source_path+typ+'*.jpg'
	files = glob.glob(file_path)
	for i, f in enumerate(files):
		target_path = dest_path+f[len(source_path):]
		print(i, len(files), target_path)
		if not os.path.exists(dest_path+typ): os.makedirs(dest_path+typ)
		cropped = read_cropped(f, size=image_size, crop_size=crop_size, interp=interp)
		scipy.misc.imsave(target_path, cropped)

