import numpy as np
import argparse
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os
import csv
import collections
import cPickle as pickle

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

# parser for the input, defining the number of training epochs and an image
#parser = argparse.ArgumentParser(description = 'Epoch: ')
#parser.add_argument('-n', metavar='N', type=int, default=0)
#parser.add_argument("-i", "--image", help="path to the input image")
#args = vars(parser.parse_args())
#epochs_id = args['n']
#image = args['image']


image_path = "/home/ersy/Downloads/aeroplane_example7.jpg"
loaded_image = image.load_img(image_path, False)



number_of_actions = 5
history_length = 8
Q_net_input_size = (25128, )


### VGG16 model without top
vgg16_conv = VGG16(include_top=False, weights='imagenet')

weights_path = '/media/ersy/Other/Google Drive/QM Work/Queen Mary/Course/Final Project/project_code/network_weights/final_weights/'

# change the weights loaded for Q network testing
saved_weights = 'Aeroplane_TEST.hdf5'
weights = weights_path+saved_weights

Q_net = reinforcement_helper.get_q_network(shape_of_input=Q_net_input_size, number_of_actions=number_of_actions, weights_path=weights)

### Q network definition
epsilon = 0
T = 60


# convert image to array	
original_image = np.array(loaded_image)
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
		print('Time Step: ', t)
		# add the current state to the experience list
		all_proposals.append(boundingbox)

		# plug state into Q network
		Q_vals = Q_net.predict(state_vec)

		action = np.argmax(Q_vals)


		if action != number_of_actions-1:
			image_copy, boundingbox = action_functions.crop_image(original_image, boundingbox, action)
		else:
			print("This is your object!")

			break

		# update history vector
		history_vec[:, :-1] = history_vec[:,1:]
		history_vec[:,-1] = [0,0,0,0,0] # hard coded actions here
		history_vec[action, -1] = 1

		preprocessed_image = image_actions.image_preprocessing(image_copy)
		state_vec = reinforcement_helper.get_state_as_vec(preprocessed_image, history_vec, vgg16_conv)




# Plotting
fig, ax = plt.subplots(1)
ax.imshow(original_image)

num_of_proposals = len(all_proposals)
color = plt.cm.rainbow(np.linspace(0,1,num_of_proposals))

for proposal, c in zip(all_proposals, color):
    top_left = (proposal[0,1], proposal[0,0])
    width = proposal[1,1] - proposal[0,1]
    height = proposal[1,0] - proposal[0,0]
    rect = patches.Rectangle(top_left, width, height, linewidth=2, edgecolor=c, facecolor='none') # change facecolor to add fill
    ax.add_patch(rect)
rect = patches.Rectangle(top_left, width, height, linewidth=2, edgecolor='white', facecolor='none' , label='proposal')
ax.add_patch(rect)

plt.legend()
plt.show()