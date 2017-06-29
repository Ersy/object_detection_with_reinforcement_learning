import numpy as np
import argparse
import matplotlib
matplotlib.use("webagg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

from keras.applications import imagenet_utils
from keras.applications.vgg16 import preprocess_input, VGG16

from keras.callbacks import ModelCheckpoint


### Local helpers
import image_actions
import reinforcement_helper
import action_functions

### 
from keras import backend as K
K.set_image_dim_ordering('tf')


### Vars
VOC_path = "/media/ersy/DATA/Google Drive/QM Work/Queen Mary/Course/Final Project/Reinforcement learning/VOCdevkit/VOC2007"



# parser for the input, defining the number of training epochs and an image
parser = argparse.ArgumentParser(description = 'Epoch: ')
parser.add_argument('-n', metavar='N', type=int, default=0)
parser.add_argument("-i", "--image", help="path to the input image")
args = vars(parser.parse_args())
epochs_id = args['n']
image = args['image']



### loading up VOC images of a given class
img_name_list = image_actions.get_img_names(VOC_path, 'aeroplane_trainval')
img_list = image_actions.load_images(VOC_path, img_name_list) 


number_of_actions = 6
history_length = 8
Q_net_input_size = (25136, )


### VGG16 model without top
vgg16_conv = VGG16(include_top=False, weights='imagenet')

Q_net = reinforcement_helper.get_q_network(shape_of_input=Q_net_input_size, number_of_actions=number_of_actions, weights_path='0')

# setting up callback to save best model
filepath="best_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='mse', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


### Q network definition
episodes = 15

# random action probability
epsilon = 0.9

# discount factor for future rewards
gamma = 0.9

# set the number of experiences to train from for each episode and batch size
number_of_experiences_to_train_from = 1000
batch_size = 100
training_iterations = number_of_experiences_to_train_from/batch_size

training_epochs = 10

# loop through images
for episode in range(episodes):

	# list to store experiences, new one for each episode (run through all images with a set epislon value)
	experiences = []

	# change the exploration-eploitation tradeoff as the episode count increases (0.9 to 0.1)
	if epsilon > 0.1:
		epsilon = epsilon -  0.2

	# iteration through all images in the image list
	for image_ix in range(5):#len(img_list)):
		

		# get initial parameters for each image
		original_image = np.array(img_list[image_ix])
		image = np.array(img_list[image_ix])
		image_name = img_name_list[image_ix]
		image_dimensions = image.shape[:-1]

		# collect bounding boxes for each image
		ground_image_bb_gt = image_actions.get_bb_gt(image_name)

		# initial bounding box (whole image, raw size)
		boundingbox = np.array([[0,0],image_dimensions])


		# list to store IOU for each object in the image and current bounding box
		IOU_list = []

		image_IOU = []
		# get the IOU for each object
		for ground_truth in ground_image_bb_gt[1]:
			current_iou = reinforcement_helper.IOU(ground_truth, boundingbox)
			image_IOU.append(current_iou)
		IOU_list.append(image_IOU)

		# create the history vector
		history_vec = np.zeros((number_of_actions, history_length))

		# preprocess the image
		preprocessed_image = image_actions.image_preprocessing(original_image)

		# get the state vector (conv output of VGG16 concatenated with the action history)
		state_vec = reinforcement_helper.get_state_as_vec(preprocessed_image, history_vec, vgg16_conv)

		# dumb trick to separate experiences for each image
		experiences.append([])

		T = 5
		for t in range(T):

			# add the current state to the experience list
			experiences[image_ix].append([state_vec])

			# plug state into Q network
			Q_vals = Q_net.predict(state_vec)

			# select the action based on the highest Q value
			best_action = np.argmax(Q_vals)

			# if the IOU is greater than 0.6 force the action to be the terminal action
			# this is done to help speed up the training process
			if max(image_IOU) > 0.6:
				best_action = 5

			# exploration or exploitation
			if random.uniform(0,1) < epsilon:
				action = best_action
				
			else:
				action = random.randint(0, 5)

			if action != 5:
				image, boundingbox = action_functions.crop_image(original_image, boundingbox, action)
			else:
				# actions to take if the trigger function is called
				print('TRIGGERED!')
				print('IOU:', max(image_IOU))


			# measure IOU
			image_IOU = []
			for ground_truth in ground_image_bb_gt[1]:
				current_iou = reinforcement_helper.IOU(ground_truth, boundingbox)
				image_IOU.append(current_iou)
			IOU_list.append(image_IOU)

			# get reward if termination action is taken
			reward = reinforcement_helper.get_reward(action, IOU_list, t)

			# get the next state

			# update history vector
			history_vec[:, :-1] = history_vec[:,1:]
			history_vec[:,-1] = [0,0,0,0,0,0] # hard coded actions here
			history_vec[action, -1] = 1

			preprocessed_image = image_actions.image_preprocessing(image)
			state_vec = reinforcement_helper.get_state_as_vec(preprocessed_image, history_vec, vgg16_conv)
			
			# add action, reward, and new state to the experience vector for the given image
			experiences[image_ix][t].append(action)
			experiences[image_ix][t].append(reward)
			experiences[image_ix][t].append(state_vec)


	# Actual training per given episode over a set number of experiences (training iterations)
	for train_iteration in range(training_iterations):
		# flatten the experiences list for learning
		flat_experiences = [x for l in experiences for x in l]

		# collect random batch of experiences
		random_ix = list(np.random.randint(0, len(flat_experiences), batch_size))
		random_experiences = np.array(flat_experiences)[random_ix]
		
		# calculating the Q values for the initial state
		initial_state = np.array([state[0] for state in random_experiences]).squeeze(1)
		initial_Q = Q_net.predict(initial_state, batch_size)

		# calculating the Q values for the next state
		next_state = np.array([state[3] for state in random_experiences]).squeeze(1)
		next_Q = Q_net.predict(next_state, batch_size)

		# get the reward for a given experience
		random_reward = np.expand_dims(random_experiences[:, 2], 1)

		# get the action of a given experience
		random_actions = np.expand_dims(random_experiences[:, 1], 1)
		flat_actions = [x for l in random_actions for x in l]


		# target for the current state should be the Q value of the next state - the reward (but only for the chosen action, the rest should be set to 0 - CURRENT NOT IMPLEMENTED)
		target = np.array(next_Q - random_reward)

		# discount the future reward, i.e the Q value output
		target = target*gamma

		# this takes the initial Q values for the state and replaces only the Q values for the actions that were used to the new target, else the error should be 0
		initial_Q[np.arange(len(initial_Q)), flat_actions] = target[np.arange(len(target)), flat_actions]

		Q_net.fit(initial_state, initial_Q, epochs=training_epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.2, verbose=0)