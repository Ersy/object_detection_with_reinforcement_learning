"""
import vgg again
import Q net
import image processing stuff
import action stuff



for each image
compute the state
feed state into the Q net
choose argmax of output
choose the action
repeat until trigger is called

measure IOU at end

"""
import numpy as np
import argparse
import matplotlib
matplotlib.use("webagg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

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



number_of_actions = 6
history_length = 8
Q_net_input_size = (25136, )
batch_size = 50

### VGG16 model without top
vgg16_conv = VGG16(include_top=False, weights='imagenet')

Q_net = reinforcement_helper.get_q_network(shape_of_input=Q_net_input_size, number_of_actions=number_of_actions, weights_path='/media/ersy/DATA/Google Drive/QM Work/Queen Mary/Course/Final Project/project_code/network_weights/Q_net_weights.h5')

### Q network definition
epsilon = 1


# Q_network = reinforcement_helper.get_q_network(shape_of_input=conv_output.shape[1:])

# loop through images
for image_ix in range(1):#len(img_list)):
    
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

    T = 10
    for t in range(T):

        # add the current state to the experience list
      

        # plug state into Q network
        Q_vals = Q_net.predict(state_vec)

        best_action = np.argmax(Q_vals)

        # exploration or exploitation
        if random.uniform(0,1) < epsilon:
            print('Yes')
            action = best_action
            
        else:
            print('no')
            action = random.randint(0, 5)

        if action != 5:
            image, boundingbox = action_functions.crop_image(original_image, boundingbox, action)
        else:
            print("This is your object!")
            # object_loc.append(boundingbox)

            for ground_truth in ground_image_bb_gt[1]:
                current_iou = reinforcement_helper.IOU(ground_truth, boundingbox)
            print(current_iou)
            break


        # measure IOU
        image_IOU = []
        for ground_truth in ground_image_bb_gt[1]:
            current_iou = reinforcement_helper.IOU(ground_truth, boundingbox)
            image_IOU.append(current_iou)
        IOU_list.append(image_IOU)


        # update history vector
        history_vec[:, :-1] = history_vec[:,1:]
        history_vec[:,-1] = [0,0,0,0,0,0] # hard coded actions here
        history_vec[action, -1] = 1

        preprocessed_image = image_actions.image_preprocessing(image)
        state_vec = reinforcement_helper.get_state_as_vec(preprocessed_image, history_vec, vgg16_conv)
