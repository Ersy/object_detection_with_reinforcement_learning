import numpy as np
import argparse
import matplotlib
matplotlib.use("webagg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from keras.applications import imagenet_utils
from keras.applications.vgg16 import preprocess_input, VGG16

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

# grab a test image
test_image = np.array(img_list[0])

# grab the name for the test image
test_name = img_list[0]

# get original dimensions of the test image
test_dimensions = test_image.shape[:-1]

### test code for classification
"""
classifier_model = VGG16(include_top=True, weights='imagenet') #classifier VGG16 model
prediction = classifier_model.predict(processed_batch_tensor, batch_size=batch_size_val)
P = imagenet_utils.decode_predictions(prediction)

for j in P:
	for (i, (imagenetID, label, prob)) in enumerate(j):
		print("{}. {}: {:.2f}%".format(i+1, label, prob*100))
	print("Next one!")

for im in processed_batch_tensor:
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

"""
defining the subregion actions
using 20 percent overlap

i want to do a test run with an original image then take a sequence of subregions 
e.g. centre, tl, tr, bl, br

"""



bb1 = np.array([[0,0],test_dimensions])
t1 = test_image
t2, bb2 = action_functions.crop_image(t1, bb1, 'centre')
t3, bb3 = action_functions.crop_image(t1, bb2, 'TL')
t4, bb4 = action_functions.crop_image(t1, bb3, 'TR')
t5, bb5 = action_functions.crop_image(t1, bb4, 'BR')
t6, bb6 = action_functions.crop_image(t1, bb5, 'BL')

"""
plt.figure(1)
plt.subplot(611)
plt.imshow(t1)
plt.subplot(612)
plt.imshow(t2)
plt.subplot(613)
plt.imshow(t3)
plt.subplot(614)
plt.imshow(t4)
plt.subplot(615)
plt.imshow(t5)
plt.subplot(616)
plt.imshow(t6)
plt.show()
"""


### VGG16 model without top
vgg16_conv = VGG16(include_top=False, weights='imagenet')

number_of_actions = 6
history_length = 8

### Q network definition

# Q_network = reinforcement_helper.get_q_network(shape_of_input=conv_output.shape[1:])

# loop through images
for image_ix in range(len(img_list)):
	
	# get initial parameters for each image
	image = np.array(img_list[image_ix])
	image_name = img_name_list[image_ix]
	image_dimensions = image.shape[:-1]

	# collect bounding boxes for each image
	ground_image_bb_gt = image_actions.get_bb_gt(image_name)

	# initial bounding box (whole image, raw size)
	initial_boundingbox = np.array([[0,0],image_dimensions])

	IOU_list = []

	for ground_truth in ground_image_bb_gt[1]:
		current_iou = reinforcement_helper.IOU(ground_truth, initial_boundingbox)
		IOU_list.append(np.array(current_iou))


	history_vec = np.array((number_of_actions, history_length))

	T = 10
	for t in range(T):


		# process the batch of images 
		preprocessed_image = image_actions.image_preprocessing(image)

		#state_vec = get_state_as_vec(preprocessed_image, history_vec, vgg16_conv)

	# plug state into Q network



		# run region through conv net (batch of images after resizing)
		# add on the actions to the conv output
		# run output state through Q network

		# choose action with the highest Q or random action 
		# (maybe one that is known to move closer)
		# update action history

		# calculate the reward
			# measure IOU before and after, if IOU is improved, score = 1
			# if action is trigger, then measure IOU, if IOU is above threshold = 3
		# calculate the new state
			# conv net + action history
		# save it all as the experience (old state, action, reward, new state

		# randomly sample a number of experiences and use for training

		# next time step