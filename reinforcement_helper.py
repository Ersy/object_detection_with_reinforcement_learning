from keras.models import Sequential # part to build the mode
from keras.layers.core import Dense, Dropout, Activation, Flatten # types of layers and associated functions
from keras.optimizers import RMSprop, SGD, Adam #optimising method (cost function and update method)
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.initializers import normal, identity

import numpy as np

# Visual descriptor size
visual_descriptor_size = 25088
# Different actions that the agent can do
number_of_actions = 5

# Number of actions in the past to retain
past_action_val = 8

movement_reward = 1
terminal_reward = 3
iou_threshold = 0.6


def get_reward(action, IOU_list, t):
	"""
	generates the correct reward based on the result of the chosen action
	"""
	if action == number_of_actions-1:
		if max(IOU_list[t+1]) > iou_threshold:
			return terminal_reward
		else:
			return -terminal_reward

	else:
		current_IOUs = IOU_list[t+1]
		past_IOUs = IOU_list[t]
		current_target = np.argmax(current_IOUs)
		if current_IOUs[current_target] - past_IOUs[current_target] > 0:
			return movement_reward
		else:
			return -movement_reward
		


def conv_net_out(image, model_vgg):
	return model_vgg.predict(image) 


### get the state by vgg_conv output, vectored, and stack on action history
def get_state_as_vec(image, history_vector, model_vgg):
	descriptor_image = conv_net_out(image, model_vgg)
	descriptor_image = np.reshape(descriptor_image, (visual_descriptor_size, 1))
	history_vector = np.reshape(history_vector, (number_of_actions*past_action_val, 1))
	state = np.vstack((descriptor_image, history_vector)).T
	return state


def get_q_network(shape_of_input, number_of_actions, weights_path='0'):
	model = Sequential()
	model.add(Dense(1024, init='lecun_uniform', input_shape = shape_of_input))# shape, name: normal(shape, scale=0.01, name=name)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1024, init='lecun_uniform'))# shape, name: normal(shape, scale=0.01, name=name)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(number_of_actions, init='lecun_uniform'))#lambda shape, name: normal(shape, scale=0.01, name=name)))
	model.add(Activation('linear'))
	adam = Adam(lr=1e-6)
	model.compile(loss='mse', optimizer=adam)
	if weights_path != "0":
		model.load_weights(weights_path)
	return model


def IOU(bb, bb_gt):
	"""
	Calculates the intersection-over-union for two bounding boxes
	"""
	x1 = max(bb[0,1], bb_gt[0,1])
	y1 = max(bb[0,0], bb_gt[0,0])
	x2 = min(bb[1,1], bb_gt[1,1])
	y2 = min(bb[1,0], bb_gt[1,0])

	w = x2-x1+1
	h = y2-y1+1

	inter = w*h
	
	aarea = (bb[1,1]-bb[0,1]+1) * (bb[1,0]-bb[0,0]+1)
	
	barea = (bb_gt[1,1]-bb_gt[0,1]+1) * (bb_gt[1,0]-bb_gt[0,0]+1)
	# intersection over union overlap
	iou = np.float32(inter) / (aarea+barea-inter)
	# set invalid entries to 0 iou - occurs when there is no overlap in x and y
	if iou < 0 or iou > 1:
		return 0
	return iou

