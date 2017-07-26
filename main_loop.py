import numpy as np
import argparse
import csv
import time
import random

from keras.applications import imagenet_utils
from keras.applications.vgg16 import preprocess_input, VGG16

from keras.callbacks import ModelCheckpoint
import cPickle

### Local helpers
import image_actions
import reinforcement_helper
import action_functions_v2 as action_functions
import get_correct_class_test
import image_augmentation
import dunno

### 
from keras import backend as K
K.set_image_dim_ordering('tf')

# parser for the input, defining the number of training epochs and an image
parser = argparse.ArgumentParser(description = 'Epoch: ')
parser.add_argument('-n', metavar='N', type=int, default=0)
parser.add_argument("-i", "--image", help="path to the input image")
args = vars(parser.parse_args())
epochs_id = args['n']
image = args['image']

project_root = '/media/ersy/Other/Google Drive/QM Work/Queen Mary/Course/Final Project/'
VOC2007_path = project_root+ 'Reinforcement learning/VOCdevkit/VOC2007'
VOC2012_path = project_root+ 'Reinforcement learning/VOCdevkit/VOC2012'

desired_class_set = 'aeroplane_trainval'
desired_class = 'aeroplane'

### loading up VOC2007 images of a given class
img_name_list_2007 = image_actions.get_img_names(VOC2007_path, desired_class_set)
groundtruths_2007, img_name_list_2007, groundtruths_label_2007 = get_correct_class_test.get_class_images(VOC2007_path, desired_class, img_name_list_2007)
img_list_2007 = image_actions.load_images(VOC2007_path, img_name_list_2007) 

### loading up VOC2012 images of a given class
# img_name_list_2012 = image_actions.get_img_names(VOC2012_path, desired_class_set)
# groundtruths_2012, img_name_list_2012, groundtruths_label_2012 = get_correct_class_test.get_class_images(VOC2012_path, desired_class, img_name_list_2012)
# img_list_2012 = image_actions.load_images(VOC2012_path, img_name_list_2012) 

### combine 2007 and 2012 datasets
img_list = img_list_2007#+img_list_2012
groundtruths = groundtruths_2007#+groundtruths_2012
img_name_list = img_name_list_2007#+img_name_list_2012
groundtruths_label = groundtruths_label_2007#+groundtruths_label_2012


# DEBUG: Overfitting hack
#img_list = [img_list[0]] *100
#groundtruths = [groundtruths[0]] *100

number_of_actions = 5
history_length = 8



# initialise Q network (randomly or with existing weights)
loaded_weights_name = 'newmodeltypez.hdf5'
loaded_weights = project_root+'project_code/network_weights/'+loaded_weights_name

Q_net = dunno.get_full_network(loaded_weights)
# setting up callback to save best model

saved_weights = 'newmodeltypez_2.hdf5'
filepath= project_root+'project_code/network_weights/' + saved_weights
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

### Q network definition
episodes = 20

# random action probability
epsilon = 1

# discount factor for future rewards
gamma = 0.9

# set the number of experiences to train from for each episode and batch size
number_of_experiences_to_train_from = 30
batch_size = 10

training_epochs = 1

# number of actions taken per image
T = 30

# IOU at which the terminal action is triggered (guided learning approach)
force_terminal = 0.6

# image data splits - lowers memory consumption per episode by only processing a subset at a time
# when selecting the chunk factor take into account the dataset size and number of actions taken
# with the full dataset and 40 actions, a chunk factor of 8 or so should be used
chunk_factor = 50
chunk_size = int(len(img_list)/chunk_factor)

# some metrics to collect in training
# collect the counts of actions in each episode of training
action_counts = []
avg_reward = []

# loop through images
for episode in range(episodes):
	print("this is episode:", episode)

	# collect count of actions in the episode
	action_count = [0,0,0,0,0]

	# collect the summation of rewards for an episode
	reward_summation = 0

	for chunk in range(chunk_factor):


		# list to store experiences, new one for each episode (run through all images with a set epislon value)
		experiences = []

		# change the exploration-eploitation tradeoff as the episode count increases (0.9 to 0.1)
		if epsilon > 0.11:
			epsilon = epsilon -  0.1


		# determines the offset to use when iterating through the chunk
		chunk_offset = chunk*chunk_size

		# iteration through all images in the current chunk
		for image_ix in range(chunk_offset,chunk_offset + chunk_size):
			print("image", image_ix)

			# get initial parameters for each image
			original_image = np.array(img_list[image_ix])
			image = np.array(img_list[image_ix])
			image_dimensions = image.shape[:-1]

			# collect bounding boxes for each image
			ground_image_bb_gt = groundtruths[image_ix]#image_actions.get_bb_gt(image_name)
			ground_image_gt_label = groundtruths_label[image_ix]
			
			## data augmentation -> 0.5 probability of flipping image and bounding box horizontally
			# augment = bool(random.getrandbits(1))
			# if augment:
			# 	original_image, ground_image_bb_gt = image_augmentation.flip_image(original_image, ground_image_bb_gt)
			# 	image = np.fliplr(image)

			# initial bounding box (whole image, raw size)
			boundingbox = np.array([[0,0],image_dimensions])

			# list to store IOU for each object in the image and current bounding box
			IOU_list = []

			image_IOU = []
			# get the initial IOU for each object
			for ground_truth in ground_image_bb_gt:
				current_iou = reinforcement_helper.IOU(ground_truth, boundingbox)
				image_IOU.append(current_iou)
			IOU_list.append(image_IOU)

			# create the history vector
			history_vec = np.zeros((number_of_actions, history_length))

			# preprocess the image
			preprocessed_image = image_actions.image_preprocessing(original_image)

			# get the state vector (conv output of VGG16 concatenated with the action history)
			state_vec = reinforcement_helper.get_state(preprocessed_image, history_vec)

			# dumb trick to separate experiences for each image
			experiences.append([])

			
			for t in range(T):

				# add the current state to the experience list
				experiences[image_ix - chunk_offset].append([state_vec])



				# exploration or exploitation
				if random.uniform(0,1) < epsilon:
		   
					# adding apprenticeship learning step - only positive actions are chosen
					good_actions = []

					for act in range(number_of_actions-1):
						potential_image, potential_boundingbox = action_functions.crop_image(original_image, boundingbox, act)            
						potential_image_IOU = []
						for ground_truth in ground_image_bb_gt:
							potential_iou = reinforcement_helper.IOU(ground_truth, potential_boundingbox)
							potential_image_IOU.append(potential_iou)
						if max(potential_image_IOU) >= max(image_IOU):
							good_actions.append(act)
					if len(good_actions) > 0:
						good_actions.append(number_of_actions-1)
						action = random.choice(good_actions)
					else:
						action = random.randint(0, number_of_actions-1)

					
				# if the IOU is greater than 0.5 force the action to be the terminal action
				# this is done to help speed up the training process
				elif max(image_IOU) > force_terminal:
					action = number_of_actions-1
				else:
					# plug state into Q network
					Q_vals = Q_net.predict(state_vec)
					# select the action based on the highest Q value
					action = np.argmax(Q_vals[0])


				# if in training the termination action is used no need to get the subcrop again
				if action != number_of_actions-1:
					image, boundingbox = action_functions.crop_image(original_image, boundingbox, action)


				# measure IOU
				image_IOU = []
				for ground_truth in ground_image_bb_gt:
					current_iou = reinforcement_helper.IOU(ground_truth, boundingbox)
					image_IOU.append(current_iou)
				IOU_list.append(image_IOU)

				# get reward if termination action is taken
				reward, target_label = reinforcement_helper.get_reward(action, IOU_list, t, ground_image_gt_label)

				# update history vector
				history_vec[:, :-1] = history_vec[:,1:]
				history_vec[:,-1] = [0,0,0,0,0] # hard coded actions here
				history_vec[action, -1] = 1

				preprocessed_image = image_actions.image_preprocessing(image)
				state_vec = reinforcement_helper.get_state(preprocessed_image, history_vec)
				
				# turn target label into binary array encoding, hard coding number of classes
				target_arr = np.zeros(20)
				target_arr[target_label-1] = 1

				# add action, reward, and new state to the experience vector for the given image
				experiences[image_ix-chunk_offset][t].append(action)
				experiences[image_ix-chunk_offset][t].append(reward)
				experiences[image_ix-chunk_offset][t].append(target_arr)
				experiences[image_ix-chunk_offset][t].append(state_vec)

				# increment the action used
				action_count[action] += 1
				reward_summation += reward

		# Actual training per given episode over a set number of experiences (training iterations)

		# flatten the experiences list for learning
		random_experiences = [x for l in experiences for x in l]

		random_experiences = np.array(random_experiences)

		# calculating the Q values for the initial state
		initial_state_im = np.array([state[0][0] for state in random_experiences]).squeeze(1)
		initial_state_hist = np.array([state[0][1] for state in random_experiences]).squeeze(1)
		initial_Q = Q_net.predict([initial_state_im,initial_state_hist], number_of_experiences_to_train_from, verbose=1)
		initial_Q = initial_Q[0]

		# calculating the Q values for the next state
		next_state_im = np.array([state[4][0] for state in random_experiences]).squeeze(1)
		next_state_hist = np.array([state[4][1] for state in random_experiences]).squeeze(1)
		next_Q_max = Q_net.predict([next_state_im,next_state_hist], number_of_experiences_to_train_from, verbose=1)

		# calculating the maximum Q for the next state
		next_Q_max = next_Q_max[0].max(axis=1)

		# get the reward for a given experience
		random_reward = random_experiences[:, 2]

		# get the action of a given experience
		random_actions = np.expand_dims(random_experiences[:, 1], 1)
		random_actions = [x for l in random_actions for x in l]

		# get the label of a given experience
		random_labels = random_experiences[:, 3]
		random_labels = np.vstack(random_labels)

		# collect the indexes of terminal actions and set next state Q value to 0
		# if the terminal action is selected the episode ends and there should be no additional reward
		terminal_indices = [i for i, x in enumerate(random_actions) if x == number_of_actions-1]
		next_Q_max[terminal_indices] = 0

		# discount the future reward, i.e the Q value output
		target = np.array(next_Q_max) * gamma

		# target for the current state should be the Q value of the next state - the reward 
		target = target + random_reward

		# repeat the target array to the same size as the initial_Q array (allowing the cost to be limited to the selected actions)
		target_repeated = np.matlib.repmat(target, 5, 1).T

		# this takes the initial Q values for the state and replaces only the Q values for the actions that were used to the new target, else the error should be 0
		initial_Q[np.arange(len(initial_Q)), random_actions] = target_repeated[np.arange(len(target_repeated)), random_actions]

		before = time.time()
		Q_net.fit([initial_state_im, initial_state_hist], [initial_Q, random_labels], epochs=training_epochs, batch_size=batch_size, shuffle=True, verbose=1, callbacks=callbacks_list, validation_split=0.2)
		after = time.time()
		print("Time taken =", after-before)

	# collect the counts of actions taken per episode
	action_counts.append(action_count)
	avg_reward.append(float(reward_summation)/len(img_list))

Q_net.save_weights('/media/ersy/Other/Google Drive/QM Work/Queen Mary/Course/Final Project/project_code/network_weights/no_validation/'+saved_weights)


# Log of training parameters
log_location = project_root + 'project_code/network_weights/logs/'

log_names = ['loaded_weights','episodes', 'epsilon','gamma', 
				'Time_steps', 'movement_reward', 'terminal_reward_5', 'terminal_reward_7', 'terminal_reward_9',
				'iou_threshold_5', 'iou_threshold_7','iou_threshold_9','update_step', 'experiences_trained_over', 'force_terminal']

log_vars = [loaded_weights, episodes, epsilon, gamma, T,reinforcement_helper.movement_reward,
			reinforcement_helper.terminal_reward_5,reinforcement_helper.terminal_reward_7,reinforcement_helper.terminal_reward_9,
			reinforcement_helper.iou_threshold_5, reinforcement_helper.iou_threshold_7,reinforcement_helper.iou_threshold_9,
			action_functions.update_step, number_of_experiences_to_train_from, force_terminal]

with open(log_location+saved_weights + '.csv', 'wb') as csvfile:
	details = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	details.writerow(log_names)	
	details.writerow(log_vars)