import numpy as np
import argparse
import matplotlib
#matplotlib.use("webagg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os
import csv
import collections

from keras.applications import imagenet_utils
from keras.applications.vgg16 import preprocess_input, VGG16

### Local helpers
import image_actions
import reinforcement_helper
import action_functions_v2 as action_functions
import get_correct_class_test

### 
from keras import backend as K
K.set_image_dim_ordering('tf')

### Vars
project_root = '/media/ersy/Other/Google Drive/QM Work/Queen Mary/Course/Final Project/'
VOC_path = project_root+ 'Reinforcement learning/VOCdevkit/VOC2007'

# parser for the input, defining the number of training epochs and an image
parser = argparse.ArgumentParser(description = 'Epoch: ')
parser.add_argument('-n', metavar='N', type=int, default=0)
parser.add_argument("-i", "--image", help="path to the input image")
args = vars(parser.parse_args())
epochs_id = args['n']
image = args['image']


### loading up VOC images of a given class
class_file = 'aeroplane_test'
img_name_list = image_actions.get_img_names(VOC_path, class_file)
img_list = image_actions.load_images(VOC_path, img_name_list) 

desired_class = 'aeroplane'

img_list, groundtruths, img_name_list = get_correct_class_test.get_class_images(VOC_path, desired_class, img_name_list, img_list)


# DEBUG: Overfitting hack
#img_list = [img_list[0]] *2
#groundtruths = [groundtruths[0]] *2

number_of_actions = 5
history_length = 8
Q_net_input_size = (25128, )

### VGG16 model without top
vgg16_conv = VGG16(include_top=False, weights='imagenet')

# path for non validated set
weights_path = '/media/ersy/Other/Google Drive/QM Work/Queen Mary/Course/Final Project/project_code/network_weights/no_validation/'

#weights_path = '/media/ersy/Other/Google Drive/QM Work/Queen Mary/Course/Final Project/project_code/network_weights/'

# change the weights loaded for Q network testing
saved_weights = 'model4.hdf5'
weights = weights_path+saved_weights

Q_net = reinforcement_helper.get_q_network(shape_of_input=Q_net_input_size, number_of_actions=number_of_actions, weights_path=weights)

### Q network definition
epsilon = 0

# stores proposal regions
all_proposals = []

# stores ground truth regions
all_ground_truth = []

all_IOU = []

all_actions = []

# IOU for terminal actions - for use in calulating evaluation stats
terminal_IOU = []
terminal_index = []

# loop through images
for image_ix in range(len(img_list)):
	
	print("new image: ", image_ix)
	# get initial parameters for each image
	original_image = np.array(img_list[image_ix])
	image = np.array(img_list[image_ix])
	image_name = img_name_list[image_ix]
	image_dimensions = image.shape[:-1]

	# collect bounding boxes for each image
	ground_image_bb_gt = groundtruths[image_ix]

	# add current image ground truth to all ground truths
	all_ground_truth.append(ground_image_bb_gt)

	# collect proposal bounding boxes
	boundingboxes = []

	#add image proposals to list of all proposals
	all_proposals.append(boundingboxes)

	# initial bounding box (whole image, raw size)
	boundingbox = np.array([[0,0],image_dimensions])

	# list to store IOU for each object in the image and current bounding box
	IOU_list = []

	# list to store actions taken for each image to associate with IOUs
	# the first IOU is associated with no action
	action_list = []
	
	image_IOU = []
	# get the IOU for each object
	for ground_truth in ground_image_bb_gt:
		current_iou = reinforcement_helper.IOU(ground_truth, boundingbox)
		image_IOU.append(current_iou)
	IOU_list.append(image_IOU)

	# create the history vector
	history_vec = np.zeros((number_of_actions, history_length))

	# preprocess the image
	preprocessed_image = image_actions.image_preprocessing(original_image)

	# get the state vector (conv output of VGG16 concatenated with the action history)
	state_vec = reinforcement_helper.get_state_as_vec(preprocessed_image, history_vec, vgg16_conv)

	T = 50
	for t in range(T):

		# add the current state to the experience list
		all_proposals[image_ix].append(boundingbox)

		# plug state into Q network
		Q_vals = Q_net.predict(state_vec)

		best_action = np.argmax(Q_vals)

	   # exploration or exploitation
		if random.uniform(0,1) < epsilon:
			action = random.randint(0, number_of_actions-1)
		else:
			action = best_action

		print('action:', action)

		if action != number_of_actions-1:
			image, boundingbox = action_functions.crop_image(original_image, boundingbox, action)
		else:
			print("This is your object!")


			current_image_IOU = []
			for ground_truth in ground_image_bb_gt:
				current_iou = reinforcement_helper.IOU(ground_truth, boundingbox)
				current_image_IOU.append(current_iou)
			print("IOU: ", current_iou)

			terminal_IOU.append(max(current_image_IOU))
			terminal_index.append(image_ix)
			action_list.append(action)
			all_actions.append(action_list)

			# implement something to mask the region covered by the boundingbox
			# rerun for the image 
			# image[boundingbox[0,0]:boundingbox[1,0], boundingbox[0,1]:boundingbox[1,1]] = mask

			break

		# measure IOU
		image_IOU = []
		for ground_truth in ground_image_bb_gt:
			current_iou = reinforcement_helper.IOU(ground_truth, boundingbox)
			image_IOU.append(current_iou)
		IOU_list.append(image_IOU)

		action_list.append(action)

		# update history vector
		history_vec[:, :-1] = history_vec[:,1:]
		history_vec[:,-1] = [0,0,0,0,0] # hard coded actions here
		history_vec[action, -1] = 1

		preprocessed_image = image_actions.image_preprocessing(image)
		state_vec = reinforcement_helper.get_state_as_vec(preprocessed_image, history_vec, vgg16_conv)

	# add the IOU calculated for each proposal for each image for evaluation purposes
	all_IOU.append(IOU_list)
	all_actions.append(action_list)



### EVALUATION AND METRICS

# lets the proposals and ground truth bounding boxes be visualised
ix = 0
image_actions.view_results(img_list, all_ground_truth, all_proposals, ix)


# simple evaluation metric
detected = sum([i>=0.5 for i in terminal_IOU])
termination_total = float(len(terminal_IOU))
termination_accuracy = detected/termination_total
print("termination accuracy = ", termination_accuracy)

flat_objects = [x for l in groundtruths for x in l]
total_objects = float(len(flat_objects))
total_accuracy = detected/total_objects
print('total accuracy = ', total_accuracy)

# obtain the accuracy for the final proposal bounding box (regardless of whether the terminal action is triggered)
final_proposal_IOU = [max(i[-1]) for i in all_IOU]
final_proposal_detected = sum([i>0.5 for i in final_proposal_IOU])
final_proposal_accuracy = final_proposal_detected/total_objects
print('final proposal accuracy = ', final_proposal_accuracy)


# turn list of IOUs for each image into separate object IOUs
t1 = [[list(j) for j in zip(*i)] for i in all_IOU]
t2 = [i for j in t1 for i in j]


fig, ax = plt.subplots(3, 1)
# code for investigating actions taken for different images - assessing the agent performance
IOU_above_cutoff = [i for i in t2 if any(j[0]>=0.5 and j[-1] >= 0.5 for j in i)]
IOU_below_cutoff = [i for i in t2 if all(j[0]<0.5 for j in i)]
for img in IOU_above_cutoff:
	ax[0].plot(img)
	ax[0].set_xlabel('action number')
	ax[0].set_ylabel('IOU')
ax[0].set_title('IOU above cutoff')

#for img in IOU_below_cutoff:
#	ax[1].plot(img)
#	ax[1].set_xlabel('action number')
#	ax[1].set_ylabel('IOU')
#ax[1].set_title('IOU below cutoff')


# storing the number of actions taken before the terminal action
action_count = [len(i) for i in all_actions if i[-1] == 4]
action_count_mean = sum(action_count)/len(action_count)
counter = collections.Counter(action_count)


ax[2].bar(counter.keys(), counter.values())
ax[2].set_xlabel("Number of actions taken")
ax[2].set_ylabel("Count")
ax[2].axvline(action_count_mean, color='red', linewidth=2)
align = 'left'
ax[2].annotate('Mean: {:0.2f}'.format(action_count_mean), xy=(action_count_mean, 1), xytext=(15, 15),
        xycoords=('data', 'axes fraction'), textcoords='offset points',
        horizontalalignment=align, verticalalignment='center',
        arrowprops=dict(arrowstyle='-|>', fc='black', shrinkA=0, shrinkB=0,
                        connectionstyle='angle,angleA=0,angleB=90,rad=10'))

plt.show()

# calculating mAP
# true positive -> IOU over 0.5 + terminal action
# false positive -> IOU under 0.5 + terminal action
# false negative -> no terminal action taken

TP = sum([i>=0.5 for i in terminal_IOU])
FP = sum([i<0.5 for i in terminal_IOU])
FN = total_objects-(TP+FP)

AP = float(TP)/(TP+FP)
Recall = float(TP)/(TP+FN)
F1 = AP*Recall/(AP+Recall)*2

print('precision = ', AP)
print('recall = ', Recall)
print('F1 = ', F1)

average_terminal_IOU = sum(terminal_IOU)/len(terminal_IOU)
print("average terminal IOU = ", average_terminal_IOU)
average_TP_IOU = sum([i for i in terminal_IOU if i>=0.5])/TP if TP >0 else 0
print("average TP IOU = ", average_TP_IOU)
average_FP_IOU = sum([i for i in terminal_IOU if i<0.5])/FP if FP>0 else 0
print("average FP IOU = ", average_FP_IOU)

###
# Get examples of images that did not have terminal actions
# Get examples of images that had a terminal IOU below 0.5
terminal_IOU_index = zip(terminal_index, terminal_IOU)
false_pos_list = [i[0] for i in terminal_IOU_index if i[1] < 0.5]


# Assessing the quality of the agent
# look at cumulative reward as a function of steps 
# calculate the reward in testing with different models
# calculate expected return

IOU_difference = [[k-j for j,k in zip(i[:-1], i[1:])] for i in t2]



# Log of parameters and testing scores
log_names = ['class_file', 'Time_steps', 'termination_accuracy', 
			'total_accuracy', 'precision', 'recall', 'F1', 'average_terminal_IOU',
			'average_TP_IOU', 'average_FP_IOU']

log_vars = [class_file, T, termination_accuracy, total_accuracy, AP, Recall, F1, 
			average_terminal_IOU, average_TP_IOU, average_FP_IOU]

log_location = project_root + 'project_code/network_weights/logs/'
with open(log_location+saved_weights + '.csv', 'a') as csvfile:
	details = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	details.writerow(log_names)	
	details.writerow(log_vars)
