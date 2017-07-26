from keras.applications import imagenet_utils
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import Sequential # part to build the mode
from keras.layers.core import Dense, Dropout, Activation, Flatten # types of layers and associated functions
from keras.optimizers import RMSprop, SGD, Adam #optimising method (cost function and update method)
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.initializers import normal, identity
from keras.models import Model
from keras.layers import Input
import keras

import numpy as np


number_of_actions = 5
past_action_val = 8

def get_full_network(weights_path=False):

	# base VGG16 model
	vgg16 = VGG16(include_top=True, weights='imagenet')
	pool=vgg16.get_layer('block5_pool').output
	pool_flat = Flatten()(pool)

	# create the classifier head
	y = Dense(1024, activation='relu')(pool_flat)
	y = Dropout(0.2)(y)
	y = Dense(1024, activation='relu')(y)
	y = Dropout(0.2)(y)
	classification_layer = Dense(20, activation='softmax', name="class_output")(y)

	# create history input
	history_input = Input(shape=(40,), dtype='float32', name='history_input')

	# create the Q network
	q_in = keras.layers.concatenate([history_input, classification_layer, pool_flat])
	x = Dense(1024, activation='relu')(q_in)
	x = Dropout(0.2)(x)
	x = Dense(1024, activation='relu')(x)
	x = Dropout(0.2)(x)
	main_output = Dense(number_of_actions, activation='linear', name="main_output")(x)


	# create and compile the model
	full_model = Model(inputs=[vgg16.input, history_input], outputs=[main_output, classification_layer])

	# freeze training in certain layers
	for layer in full_model.layers[:15]:
		layer.trainable = False


	adam = Adam(lr=1e-6)
	full_model.compile(optimizer=adam,
	              loss={'main_output': 'mse', 'class_output': 'categorical_crossentropy'},
	              loss_weights={'main_output': 1., 'class_output': 0.2})
	
	# load in saved weights
	if weights_path:
		full_model.load_weights(weights_path)
	
	return full_model

full_model = get_full_network()


# test stuff to try it out
if __name__ == "__main__":
	# Creating some fake input data
	# image data
	im = np.random.rand(1,224,224,3)
	im_100 = np.tile(im, (100,1,1,1))

	#  history data
	history_vector = np.zeros((number_of_actions, past_action_val))
	action_hist = np.reshape(history_vector, (number_of_actions*past_action_val, 1))
	a = action_hist.squeeze()
	b = np.expand_dims(a, axis=0)
	b_100 = np.tile(b, (100,1))

	# Creating fake target data
	# movement data
	move_target = np.expand_dims(np.array([0,0,0,0,1]).squeeze(), axis=0)
	move_target_100 = np.tile(move_target, (100,1))

	# class data
	class_target = np.expand_dims(np.zeros(20).squeeze(), axis=0)
	class_target[0][0] = 1
	class_target_100 = np.tile(class_target, (100,1))


	# run a prediction before training
	out = full_model.predict([im, b])
	print("before:", out)

	# train it
	full_model.fit({'input_1': im_100, 'history_input': b_100},
	          {'main_output': move_target_100, 'class_output': class_target_100},
	          epochs=50, batch_size=20)

	# run a prediction after training
	out = full_model.predict([im, b])
	print("after:", out)