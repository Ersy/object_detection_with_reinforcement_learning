from keras.models import Sequential # part to build the mode
from keras.layers.core import Dense, Dropout, Activation, Flatten # types of layers and associated functions
from keras.optimizers import RMSprop, SGD, Adam #optimising method (cost function and update method)
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.initializers import normal, identity

import numpy as np

"""
### get the state by vgg_conv output, vectored, and stack on action history
def get_state(image, history_vector, model_vgg):
	descriptor_image = get_conv_image_descriptor_for_image(image, model_vgg)
	descriptor_image = np.reshape(descriptor_image, (visual_descriptor_size, 1))
	history_vector = np.reshape(history_vector, (number_of_actions*actions_of_history, 1))
	state = np.vstack((descriptor_image, history_vector))
	return state
"""

def get_q_network(shape_of_input, weights_path='0'):
	model = Sequential()
	model.add(Flatten(input_shape=shape_of_input))
	model.add(Dense(1024, init='lecun_uniform'))# shape, name: normal(shape, scale=0.01, name=name)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1024, init='lecun_uniform'))# shape, name: normal(shape, scale=0.01, name=name)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(4, init='lecun_uniform'))#lambda shape, name: normal(shape, scale=0.01, name=name)))
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
	yA = max(bb[0,0], bb_gt[0,0])
	xA = max(bb[0,1], bb_gt[0,1])
	yB = min(bb[1,0], bb_gt[1,0])
	xB = min(bb[1,1], bb_gt[1,1])

	interArea = (xB - xA + 1) * (yB - yA + 1)

	bb_Area = (bb[1,0] - bb[0,0] + 1) * (bb[1,1] - bb[0,1] + 1)
	bb_gt_Area = (bb_gt[1,0] - bb_gt[0,0] + 1) * (bb_gt[1,1] - bb_gt[0,1] + 1)

	iou = interArea / float(bb_Area + bb_gt_Area - interArea)
	return iou
