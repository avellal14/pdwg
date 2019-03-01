import glob
import scipy
import scipy.misc
import pdb 
import os

def read_cropped(filename, size, interp):
	rgb = scipy.misc.imread(filename)
	rgb_scaled = scipy.misc.imresize(rgb, (size, size), interp=interp)
	return rgb_scaled

# image_size = 64
image_size = 128
# interp='nearest'
# interp='bilinear'
# interp='bicubic'
interp='cubic'
# interp='lanczos'

# source_path = '/home/mcgemici/cat_data/cats_bigger_than_64x64/'
source_path = '/home/mcgemici/cat_data/cats_bigger_than_128x128/'

dest_path = '/home/mcgemici/cat_data/cats_'+str(image_size)+'_'+interp+'/'
file_path = source_path+'*.jpg'
files = glob.glob(file_path)
for i, f in enumerate(files):
	target_path = dest_path+f[len(source_path):]
	print(i, len(files), target_path)
	if not os.path.exists(dest_path): os.makedirs(dest_path)
	cropped = read_cropped(f, size=image_size, interp=interp)
	scipy.misc.imsave(target_path, cropped)

