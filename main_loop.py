import numpy as np
import argparse
import cv2


from keras.applications import imagenet_utils
from keras.applications.vgg16 import preprocess_input, VGG16

### Local helpers
import image_actions
import reinforcement_helper

### 
from keras import backend as K
K.set_image_dim_ordering('tf')


### Vars
VOC_path = "/media/ersy/DATA/Google Drive/QM Work/Queen Mary/Course/Final Project/Reinforcement learning/VOCdevkit/VOC2007"
batch_size_val = 5



# parser for the input, defining the number of training epochs and an image
parser = argparse.ArgumentParser(description = 'Epoch: ')
parser.add_argument('-n', metavar='N', type=int, default=0)
parser.add_argument("-i", "--image", help="path to the input image")
args = vars(parser.parse_args())
epochs_id = args['n']
image = args['image']

print(epochs_id)


### loading up VOC images of a given class
img_name_list = image_actions.get_img_names(VOC_path, 'aeroplane_trainval')
img_list = image_actions.load_images(VOC_path, img_name_list) 


### get original dimensions of each image


### converting images of a given batch size to an image tensor 
image_batch_tensor, image_dims_list = image_actions.batch_image_preprocessing(img_list[:batch_size_val]) #expects a list of PIL objects

""" 
### test code for classification

classifier_model = VGG16(include_top=True, weights='imagenet') #classifier VGG16 model
prediction = classifier_model.predict(image_batch_tensor, batch_size=batch_size_val)
P = imagenet_utils.decode_predictions(prediction)

for j in P:
	for (i, (imagenetID, label, prob)) in enumerate(j):
		print("{}. {}: {:.2f}%".format(i+1, label, prob*100))
	print("Next one!")

for im in image_batch_tensor:
	image_actions.view_image(im)

"""

"""
### VGG16 model without top
vgg16_conv = VGG16(include_top=False, weights='imagenet')
conv_output = vgg16_conv.predict(image_batch_tensor, batch_size=batch_size_val)


### Q network definition

Q_network = reinforcement_helper.get_q_network(shape_of_input=conv_output.shape[1:])
"""

"""
The input to the Q network should be the current state, i.e. the output of the VGG conv layers +
some additional information, e.g. the past actions
"""

"""
target_test = np.array([[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,1,0,0],[0,0,1,0]])
#target_test = np.expand_dims(target_test, 0)
### training
learn_test = Q_network.fit(conv_output, target_test, batch_size=batch_size_val, nb_epoch=500, verbose=1)

print(Q_network.predict(conv_output, batch_size=batch_size_val))
"""

### Action definition
## subcrop of image (of original or resized?) - should be the original somehow, or more information loss
"""
testing out extracting a subregion from bigger image

the action will define the nature of the subcrop bounding box
the bounding box will then be used with respect to the original image to get the subcrop
the subcrop will then be processed and fed into the conv net

are all input images the same size?
no so we'll have to save the original dimensions beforehand
what a drag

"""
t1 = image_batch_tensor[0] #grab an image
bb = np.array([[10, 10],[100,100]]) #define a bounding box subregion
t2 = t1[bb[0,0]:bb[1,0], bb[0,1]:bb[1,1]] #use bounding box subregion to slice image array


"""
defining the subregion actions
using 20 percent overlap

i want to do a test run with an original image then take a sequence of subregions 
e.g. centre, tl, tr, bl, br

"""


def get_TL_bb(origin, width, height):
	x_origin = origin[0]
	y_origin = origin[1]
	top_left = [x_origin, y_origin]
	bottom_right = [int(x_origin+0.6*width), int(y_origin+0.6*height)]
	return [top_left, bottom_right]

def get_TR_bb(origin, width, height):
	x_origin = origin[0]
	y_origin = origin[1]
	top_left = [int(x_origin+0.4*width), y_origin]
	bottom_right = [width, int(y_origin+0.6*height)]
	return [top_left, bottom_right]

def get_BL_bb(origin, width, height):
	x_origin = origin[0]
	y_origin = origin[1]
	top_left = [x_origin, int(y_origin+0.4*height)]
	bottom_right = [int(x_origin+0.6*width), height]
	return [top_left, bottom_right]

def get_BR_bb(origin, width, height):
	x_origin = origin[0]
	y_origin = origin[1]
	top_left = [int(x_origin+0.4*width), int(y_origin+0.4*height)]
	bottom_right = [width, height]
	return [top_left, bottom_right]

def get_Centre_bb(origin, width, height):
	x_origin = origin[0]
	y_origin = origin[1]
	top_left = [int(x_origin+0.2*width), int(y_origin+0.2*width)]
	bottom_right = [int(x_origin+0.8*width), int(y_origin+0.8*height)]
	return [top_left, bottom_right]

def get_subcrop(im, bb, region):
	origin = bb[0]
	width = bb[1][0] - bb[0][0]
	height = bb[1][1] - bb[0][1]
	if region == 'TL':
		subregion_bb = get_TL_bb(origin, width, height)
	elif region == 'TR':
		subregion_bb = get_TR_bb(origin, width, height)
	elif region == 'BL':
		subregion_bb = get_BL_bb(origin, width, height)
	elif region == 'BR':
		subregion_bb = get_BR_bb(origin, width, height)
	elif region == 'Centre':
		subregion_bb = get_Centre_bb(origin, width, height)
	start_x = subregion_bb[0][0]
	start_y = subregion_bb[0][1]
	end_x = subregion_bb[1][0]
	end_y = subregion_bb[1][1]
	im_2 = im[start_x:end_x,start_y:end_y,:]
	return im_2, subregion_bb



t1 = image_batch_tensor[0]
bb1 = [[0,0], list(image_dims_list[0])]
t2, bb2 = get_subcrop(t1, bb1, region='TL')
t3, bb3 = get_subcrop(t2, bb2, region='TR')
t4, bb4 = get_subcrop(t3, bb3, region='BL')
t5, bb5 = get_subcrop(t4, bb4, region='BL')


### STILL BROKEN!!!!!
### t5 ends up as empty for some reason


