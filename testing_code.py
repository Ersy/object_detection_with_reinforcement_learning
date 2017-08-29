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
import cPickle as pickle

from keras.applications import imagenet_utils
from keras.applications.vgg16 import preprocess_input, VGG16

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
parser = argparse.ArgumentParser(description = 'Epoch: ')
parser.add_argument('-n', metavar='N', type=int, default=0)
parser.add_argument("-i", "--image", help="path to the input image")
args = vars(parser.parse_args())
epochs_id = args['n']
image = args['image']

VOC = True
if VOC:
	### loading up VOC images of a given class
	class_file = 'person_test'
	img_name_list = image_actions.get_img_names(VOC_path, class_file)
	img_list = image_actions.load_images(VOC_path, img_name_list) 

	desired_class = 'person'

	img_list, groundtruths, img_name_list = image_loader.get_class_images(VOC_path, desired_class, img_name_list, img_list)
else:
	class_file = 'Experiment_1'
	img_list = pickle.load(open(project_root+'project_code/pickled_data/Experiment_8_Test_images.pickle', 'rb'))
	groundtruths = pickle.load(open(project_root+'project_code/pickled_data/Experiment_8_Test_boxes.pickle', 'rb'))

# DEBUG: Overfitting hack
#img_list = img_list[0:8]
#groundtruths = groundtruths[0:8]

number_of_actions = 5
history_length = 8
Q_net_input_size = (25128, )


### VGG16 model without top
vgg16_conv = VGG16(include_top=False, weights='imagenet')

# path for non validated set
#weights_path = '/media/ersy/Other/Google Drive/QM Work/Queen Mary/Course/Final Project/project_code/network_weights/no_validation/'

weights_path = '/media/ersy/Other/Google Drive/QM Work/Queen Mary/Course/Final Project/project_code/network_weights/final_weights/'

# change the weights loaded for Q network testing
saved_weights = 'Person_TEST.hdf5'
weights = weights_path+saved_weights

Q_net = reinforcement_helper.get_q_network(shape_of_input=Q_net_input_size, number_of_actions=number_of_actions, weights_path=weights)

### Q network definition
epsilon = 0
T = 60
# stores proposal regions
all_proposals = []

# stores ground truth regions
all_ground_truth = []

all_IOU = []

all_actions = []

all_image_scale= []
all_image_centre = []

# IOU for terminal actions - for use in calulating evaluation stats
terminal_IOU = []
terminal_index = []

# loop through images
for image_ix in range(len(img_list)):
	
	original_image = np.array(img_list[image_ix])

	print("new image: ", image_ix)
	# get initial parameters for each image

	image = np.copy(original_image)
	#image_name = img_name_list[image_ix]
	image_dimensions = image.shape[:-1]

	# collect bounding boxes for each image
	ground_image_bb_gt = groundtruths[image_ix]

	# METRICS: get the scale of the object relative to the image size
	
	image_scale = []
	image_centre = []
	for box in ground_image_bb_gt:

		width = box[1][1] - box[0][1]
		height = box[1][0] - box[0][0]
		area = width*height

		image_area = image_dimensions[0]*image_dimensions[1]
		image_scale.append(float(area)/image_area)
		image_centre.append([(box[1][0] + box[0][0])/2, (box[1][1] + box[0][1])/2])
	all_image_scale.append(image_scale)
	all_image_centre.append(image_centre)

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
			print("IOU: ", max(current_image_IOU))

			terminal_IOU.append(max(current_image_IOU))
			terminal_index.append(image_ix)
			action_list.append(action)
			#all_actions.append(action_list)

			# implement something to mask the region covered by the boundingbox
			# rerun for the image 
			#mask =  [103.939, 116.779, 123.68]
			#original_image[boundingbox[0,0]:boundingbox[1,0], boundingbox[0,1]:boundingbox[1,1]] = mask

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
image_actions.view_results(img_list, all_ground_truth, all_proposals, all_IOU, ix)


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


fig, ax = plt.subplots(4, 1, sharex=True)
# code for investigating actions taken for different images - assessing the agent performance

# objects with the final IoU above 0.5 (terminal action called)
IOU_above_cutoff  = [i for i in t2 if i[-1]>=0.5]

# object 
IOU_below_cutoff = [i for i in t2 if i[-1]<0.5 and len(i) < T+1]

# objects with no terminal action called
IOU_no_terminal = [i for i in t2 if i[-1]<0.5 and len(i) == T+1]

for img in IOU_above_cutoff:
	ax[0].plot(img)
ax[0].set_ylabel('IOU')
ax[0].set_title('IOU above cutoff')
ax[0].set_ylim(0,1)

for img in IOU_below_cutoff:
	ax[1].plot(img)
ax[1].set_ylabel('IOU')
ax[1].set_title('IOU below cutoff')
ax[1].set_ylim(0,1)

for img in IOU_no_terminal:
	ax[2].plot(img)
ax[2].set_ylabel('IOU')
ax[2].set_title('IOU no terminal actions')
ax[2].set_ylim(0,1)

# storing the number of actions taken before the terminal action
action_count = [len(i) for i in all_actions if i[-1] == 4]
action_count_mean = sum(action_count)/len(action_count)
counter = collections.Counter(action_count)

ax[3].bar(counter.keys(), counter.values())
ax[3].set_xlabel("Actions taken")
ax[3].set_ylabel("Count")
ax[3].set_title('Actions per image (terminal action used)')
ax[3].axvline(action_count_mean, color='red', linewidth=2, label='MEAN: '+str(action_count_mean)[:5])
ax[3].legend()

plt.xlim(0,T)
plt.tight_layout()
plt.show()

# calculating mAP
# true positive -> IOU over 0.5 + terminal action
# false positive -> IOU under 0.5 + terminal action
# false negative -> no terminal action taken when image contains an object
# true negative -> no terminal action taken when image does not contain an object

TP = sum([i>=0.5 for i in terminal_IOU])
FP = sum([i<0.5 for i in terminal_IOU])
FN = total_objects-(TP+FP)

AP = float(TP)/(TP+FP)

if TP+FN > 0:
	Recall = float(TP)/(TP+FN)
else:
	Recall = 0

if AP > 0:
	F1 = AP*Recall/(AP+Recall)*2
else:
	F1 = 0


print('precision = ', AP)
print('recall = ', Recall)
print('F1 = ', F1)

average_terminal_IOU = sum(terminal_IOU)/len(terminal_IOU)
print("average terminal IOU = ", average_terminal_IOU)
std_terminal_IOU = np.std(terminal_IOU)
print("terminal IOU standard deviation = ", std_terminal_IOU)
average_TP_IOU = sum([i for i in terminal_IOU if i>=0.5])/TP if TP >0 else np.nan
print("average TP IOU = ", average_TP_IOU)
average_FP_IOU = sum([i for i in terminal_IOU if i<0.5])/FP if FP>0 else np.nan
print("average FP IOU = ", average_FP_IOU)

# Plot distributions of terminal IOUs
bins = np.arange(0,1,0.02)
plt.hist([i for i in terminal_IOU if i>=0.5], bins=bins, color='red')
plt.hist([i for i in terminal_IOU if i<0.5], bins=bins, color='blue')
plt.xlim(0,1)
plt.ylim(0,500)
plt.axvline(average_terminal_IOU, color='black', label='MEAN: '+ str(average_terminal_IOU)[:5])
plt.axvline(average_terminal_IOU-std_terminal_IOU, color='gray', linestyle='--', label='STDEV: '+ str(std_terminal_IOU)[:5])
plt.axvline(average_terminal_IOU+std_terminal_IOU, color='gray', linestyle='--')
plt.xlabel('IoU')
plt.ylabel('Count')
plt.legend()
plt.show()

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


from scipy.interpolate import griddata
yx = np.vstack(all_image_centre).T
y = yx[0,:]
x = yx[1,:]
z = list(np.vstack([i[-1] for i in all_IOU]).T[0])
xi = np.linspace(x.min(), x.max(), x.max()-x.min()+1)
yi = np.linspace(y.min(), y.max(), y.max()-y.min()+1)
zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')

zmin = 0.0
zmax = 1.0
zi[(zi<zmin)] = zmin
zi[(zi>zmax)] = zmax

cs = plt.contourf(xi, yi, zi, 15, cmap=plt.cm.rainbow, vmax=zmax, vmin=zmin)
plt.colorbar()
plt.show()






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


