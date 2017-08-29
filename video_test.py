import cv2
import numpy as np
import argparse

import random
import os
import csv
import collections


from keras.applications import imagenet_utils
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.preprocessing import image

### Local helpers
import image_actions
import reinforcement_helper
import action_functions
import image_loader


### 
from keras import backend as K
K.set_image_dim_ordering('tf')

### Vars
project_root = '/media/ersy/Other/Google Drive/QM Work/Queen Mary/Course/Final Project/'
VOC_path = project_root+ 'Reinforcement learning/VOCdevkit/VOC2007'


#image_path = "/home/ersy/Downloads/aeroplane_example7.jpg"
#loaded_image = image.load_img(image_path, False)



number_of_actions = 5
history_length = 8
Q_net_input_size = (25128, )


### VGG16 model without top
vgg16_conv = VGG16(include_top=False, weights='imagenet')

weights_path = '/media/ersy/Other/Google Drive/QM Work/Queen Mary/Course/Final Project/project_code/network_weights/final_weights/'

# change the weights loaded for Q network testing
saved_weights = 'Person_TEST.hdf5'
weights = weights_path+saved_weights

Q_net = reinforcement_helper.get_q_network(shape_of_input=Q_net_input_size, number_of_actions=number_of_actions, weights_path=weights)

### Q network definition
T = 60

def detectObject(original_image, T):
	"""
	takes in image as a numpy array, and a number of time steps then returns a localising bounding box around the object
	"""
	
	image_copy = np.copy(original_image)
	image_dimensions = image_copy.shape[:-1]

	# create the history vector
	history_vec = np.zeros((number_of_actions, history_length))

	# preprocess the image
	preprocessed_image = image_actions.image_preprocessing(original_image)

	# get initial state vector
	state_vec = reinforcement_helper.get_state_as_vec(preprocessed_image, history_vec, vgg16_conv)

	# get initial bounding box
	boundingbox = np.array([[0,0],image_dimensions])

	all_proposals = []

	for t in range(T):
			# add the current state to the experience list
			all_proposals.append(boundingbox)

			# plug state into Q network
			Q_vals = Q_net.predict(state_vec)

			action = np.argmax(Q_vals)


			if action != number_of_actions-1:
				image_copy, boundingbox = action_functions.crop_image(original_image, boundingbox, action)
			else:
				print("This is your object!")
				return boundingbox
				#break


			# update history vector
			history_vec[:, :-1] = history_vec[:,1:]
			history_vec[:,-1] = [0,0,0,0,0] # hard coded actions here
			history_vec[action, -1] = 1

			preprocessed_image = image_actions.image_preprocessing(image_copy)
			state_vec = reinforcement_helper.get_state_as_vec(preprocessed_image, history_vec, vgg16_conv)

	return all_proposals[-1]




cap = cv2.VideoCapture('/media/ersy/Other/Google Drive/QM Work/Queen Mary/Course/Final Project/project_code/videos/Golf_Swing.mp4')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

fc = 0
ret = True

while (fc<205):#cap.read()[0]==True):
    ret, buf[fc] = cap.read()
    fc += 1

cap.release()


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('Golf_Swing.avi',fourcc, 24.0, (frameWidth, frameHeight), isColor=True)


for frame in range(frameCount):
	print("Frame: ", frame)
	cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('frame', 600,600)
	bb = detectObject(buf[frame], 60)
	cv2.rectangle(buf[frame], (bb[0,1], bb[0,0]),(bb[1,1],bb[1,0]),(0,0,255),2)
	
	out.write(buf[frame])
	cv2.imshow('frame', buf[frame])
	cv2.waitKey(1)

#cv2.namedWindow('frame 10')
#cv2.imshow('frame 10', buf[9])

#cv2.waitKey(0)
out.release()

cv2.destroyAllWindows()
