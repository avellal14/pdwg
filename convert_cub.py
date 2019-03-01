import glob
import scipy
import scipy.misc
import pdb 
import os


main_path = '/home/mcgemici/cub_data/CUB_200_2011/'
desired_size = 64
with open(main_path+'bounding_boxes.txt') as f:
    bounding_boxes_content = f.readlines()
bounding_boxes_content = [x.strip() for x in bounding_boxes_content] 
with open(main_path+'images.txt') as f:
    images_content = f.readlines()
images_content = [x.strip() for x in images_content] 
bounding_boxes_content_split = []
for line in bounding_boxes_content: bounding_boxes_content_split.append(line.split())
images_content_split = []
for line in images_content: images_content_split.append(line.split())

aspect_valid_counter = 0
max_valid_counter = 0
min_valid_counter = 0
for fileprop in bounding_boxes_content_split:
	nameID, x1, y1, w, h = int(float(fileprop[0])), int(float(fileprop[1])), int(float(fileprop[2])), int(float(fileprop[3])), int(float(fileprop[4]))
	x2, y2 = x1+w, y1+h
	# assert(fileprop[0]== images_content_split[nameID-1][0])
	# assert(int(float(images_content_split[nameID-1][0])) == nameID)
	
	filename = images_content_split[nameID-1][1]
	full_file_path = main_path +'images/'+ filename
	image = scipy.misc.imread(full_file_path)
	if len(image.shape)!=3: continue
	bounding_box_image = image[y1:y2, x1:x2]

	write_file_path = main_path +'images_processed_aspect_'+str(desired_size)+'/'+ filename
	write_file_path_dironly = main_path +'images_processed_aspect_'+str(desired_size)+'/'+ filename.split('/')[0]+'/'
	aspect_valid_counter += 1
	new_image = scipy.misc.imresize(bounding_box_image, (desired_size, desired_size), interp='bilinear')
	if not os.path.exists(write_file_path_dironly): os.makedirs(write_file_path_dironly)
	scipy.misc.imsave(write_file_path, new_image)
	print('Success for file: ', filename)

	max_size = max(bounding_box_image.shape[0], bounding_box_image.shape[1])
	mean_pixel_x = int((float(x1)+float(x2))/2.)
	mean_pixel_y = int((float(y1)+float(y2))/2.)
	x1_new, x2_new = mean_pixel_x-int(float(max_size)/2), mean_pixel_x+int(float(max_size)/2)
	y1_new, y2_new = mean_pixel_y-int(float(max_size)/2), mean_pixel_y+int(float(max_size)/2)
	write_file_path = main_path +'images_processed_max_'+str(desired_size)+'/'+ filename
	write_file_path_dironly = main_path +'images_processed_max_'+str(desired_size)+'/'+ filename.split('/')[0]+'/'
	if x1_new<0 or y1_new<0 or x2_new>image.shape[0] or y2_new>image.shape[1]:
		print('Failure for file: ', filename)
	else:
		max_valid_counter += 1
		new_image = scipy.misc.imresize(image[y1_new:y2_new, x1_new:x2_new], (desired_size, desired_size), interp='bilinear')
		if not os.path.exists(write_file_path_dironly): os.makedirs(write_file_path_dironly)
		scipy.misc.imsave(write_file_path, new_image)
		print('Success for file: ', filename)

	min_size = min(bounding_box_image.shape[0], bounding_box_image.shape[1])
	mean_pixel_x = int((float(x1)+float(x2))/2.)
	mean_pixel_y = int((float(y1)+float(y2))/2.)
	x1_new, x2_new = mean_pixel_x-int(float(min_size)/2), mean_pixel_x+int(float(min_size)/2)
	y1_new, y2_new = mean_pixel_y-int(float(min_size)/2), mean_pixel_y+int(float(min_size)/2)
	write_file_path = main_path +'images_processed_min_'+str(desired_size)+'/'+ filename
	write_file_path_dironly = main_path +'images_processed_min_'+str(desired_size)+'/'+ filename.split('/')[0]+'/'
	if x1_new<0 or y1_new<0 or x2_new>image.shape[1] or y2_new>image.shape[0]:
		print('Failure for file: ', filename)
	else:
		min_valid_counter += 1
		new_image = scipy.misc.imresize(image[y1_new:y2_new, x1_new:x2_new], (desired_size, desired_size), interp='bilinear')
		if not os.path.exists(write_file_path_dironly): os.makedirs(write_file_path_dironly)
		scipy.misc.imsave(write_file_path, new_image)
		print('Success for file: ', filename)

print('aspect Valid images: ', aspect_valid_counter, len(bounding_boxes_content_split))
print('max Valid images: ', max_valid_counter, len(bounding_boxes_content_split))
print('min Valid images: ', min_valid_counter, len(bounding_boxes_content_split))




	# bounding_box_image = image[x1:x2, y1:y2]
	# bounding_box_image = image[-y2:-y1, x1:x2]
	# bounding_box_image = image[max(0, image.shape[0]-y2+1):image.shape[0]-y1, x1:x2]

	# max(bounding_box_image.shape)
	# ratio = float(bounding_box_image.shape[0])/float(bounding_box_image.shape[1])

	# try:
	# 	new_image = bounding_box_image
	# 	if not os.path.exists(write_file_path_dironly): os.makedirs(write_file_path_dironly)
	# 	scipy.misc.imsave(write_file_path, new_image)
	# 	print('Success for file: ', filename)
	# except:
	# 	pdb.set_trace()
