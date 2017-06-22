from PIL import Image, ImageFilter
from keras.preprocessing import image
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from keras.applications.vgg16 import preprocess_input


### Reference values
class_name_dict = { 'aeroplane':1,
					'bicycle': 2,
					'bird': 3,
					'boat':4, 
					'bottle': 5,
					'bus':6,
					'car':7,
					'cat': 8,
					'chair': 9,
					'cow':10,
					'diningtable':11,
					'dog':12,
					'horse':13,
					'motorbike': 14,
					'person':15,
					'pottedplan':16,
					'sheep':17,
					'sofa':18,
					'train':19,
					'tvmonitor':20
					}

VOC_path = "/media/ersy/DATA/Google Drive/QM Work/Queen Mary/Course/Final Project/Reinforcement learning/VOCdevkit/VOC2007"



def load_images(VOC_path, image_names):
	"""
	loads images from a given data set
	"""
	images = []
	for i in range(len(image_names)):
		image_name = image_names[i]
		string = VOC_path + '/JPEGImages/' + image_name + '.jpg'
		images.append(image.load_img(string, False))
	return images


def get_img_names(VOC_path, data_set_name):
	"""
	collects the file names associated with a class and data set type
	"""
	file_path = VOC_path + '/ImageSets/Main/' + data_set_name + '.txt'
	f = open(file_path)
	image_names = f.readlines()
	image_names = [x.strip('\n') for x in image_names]
	f.close()
	return [x.split(None, 1)[0] for x in image_names]


def get_img_labels(VOC_path, data_set_name):
	"""
	collects the labels for the desired dataset
	"""
	file_path = VOC_path + '/ImageSets/Main/' + data_set_name + '.txt'
	f = open(file_path)
	image_names = f.readlines()
	image_names = [x.strip('\n') for x in image_names]
	f.close()
	return [x.split(None, 1)[1] for x in image_names]
	

def get_bb_gt(image_name):
	"""
	get the ground truth bounding box values and class for an image
	"""
	file_path = VOC_path + '/Annotations/' + image_name + '.xml'
	tree = ET.parse(file_path)
	root = tree.getroot()
	names = []
	x_min = []
	x_max = []
	y_min = []
	y_max = []
	for child in root:
		if child.tag == 'object':
			for child2 in child:
				if child2.tag == 'name':
					names.append(child2.text)
				elif child2.tag == 'bndbox':
					for child3 in child2:
						if child3.tag == 'xmin':
							x_min.append(child3.text)
						elif child3.tag == 'xmax':
							x_max.append(child3.text)
						elif child3.tag == 'ymin':
							y_min.append(child3.text)
						elif child3.tag == 'ymax':
							y_max.append(child3.text)
	category_and_bb = np.zeros([np.size(names), 5])
	for i in range(np.size(names)):
		category_and_bb[i][0] = class_name_dict[names[i]]
		category_and_bb[i][1] = x_min[i]
		category_and_bb[i][2] = x_max[i]
		category_and_bb[i][3] = y_min[i]
		category_and_bb[i][4] = y_max[i]
	return category_and_bb


def view_image(t0):
	"""
	converts an image back into a viewable format and displays
	"""
	#t3 = np.moveaxis(t2, 0, 2) # not needed for tf backend
	#t0 = t0[:, :, :, ::-1]
	t0[:, :, 0] += 103.939
	t0[:, :, 1] += 116.779
	t0[:, :, 2] += 123.68
	t1 = np.uint8(t0)
	t2 = Image.fromarray(t1)
	t2.show()


def batch_image_preprocessing(batch_of_images):
	"""
	preprocessing for images before VGG16
	"""
	im_list =[]
	for im in batch_of_images:
		im_numpy = np.array(im)
		im_numpy = image_preprocessing(im_numpy)
		im_list.append(im_numpy)
	im_tensor = np.concatenate(im_list)
	im_batch = preprocess_input(im_tensor)
	return im_batch


def image_preprocessing(im):
	"""
	preprocessing for images before VGG16
	"""
	im_numpy = np.array(im)
	im_numpy = im_numpy[:, :, ::-1] # keep this in if the color channel order needs reversing
	### given an image, get the features (i.e. run it through the conv net)
	im = cv2.resize(im_numpy, (224, 224)).astype(np.float32)
	# reverse the ordering of the dimensions (not needed for tf backend)
	#im = np.moveaxis(im, 2, 0)
	im = np.expand_dims(im, axis=0)
	return im


if __name__ == '__main__':
	img_name_list = get_img_names(VOC_path, 'boat_trainval')
	img_list = load_images(VOC_path, img_name_list)
	#img_labels = get_img_labels(VOC_path, 'boat_train')
	img_list[0].show()