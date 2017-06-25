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

### 
from keras import backend as K
K.set_image_dim_ordering('tf')


### Vars
VOC_path = "/media/ersy/DATA/Google Drive/QM Work/Queen Mary/Course/Final Project/Reinforcement learning/VOCdevkit/VOC2007"
batch_size_val = 5



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


### get original dimensions of each image


### converting images of a given batch size to an image tensor 
raw_image_batch_list, image_dims_list = image_actions.batch_image_raw_data(img_list[:batch_size_val]) #expects a list of PIL objects
processed_batch_tensor = image_actions.batch_image_preprocessing(raw_image_batch_list)



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



def TL_bb(bb):
    """Takes a bounding box and returns a bounding box of the top left region"""
    y_origin = bb[0,0]
    x_origin = bb[0,1]
    
    y_end = bb[1,0]
    x_end = bb[1,1]

    tl = [y_origin, x_origin]
    br = [y_end*0.6, x_end*0.6]
    return np.array([tl, br])


def TR_bb(bb):
    """Takes a bounding box and returns a bounding box of the top right region"""
    y_origin = bb[0,0]
    x_origin = bb[0,1]
    
    y_end = bb[1,0]
    x_end = bb[1,1]

    tl = [y_origin, x_origin+((x_end-x_origin)*0.4)]
    br = [y_end*0.6, x_end]
    return np.array([tl, br])


def BL_bb(bb):
    """Takes a bounding box and returns a bounding box of the bottom left region"""
    y_origin = bb[0,0]
    x_origin = bb[0,1]
    
    y_end = bb[1,0]
    x_end = bb[1,1]

    tl = [y_origin+((y_end-y_origin)*0.4), x_origin]
    br = [y_end, x_end*0.6]
    return np.array([tl, br])


def BR_bb(bb):
    """Takes a bounding box and returns a bounding box of the bottom right region"""
    y_origin = bb[0,0]
    x_origin = bb[0,1]
    
    y_end = bb[1,0]
    x_end = bb[1,1]

    tl = [y_origin+((y_end-y_origin)*0.4), x_origin+((x_end-x_origin)*0.4)]
    br = [y_end, x_end]
    return np.array([tl, br])

def centre_bb(bb):
    """Takes a bounding box and returns a bounding box of the centre region"""
    y_origin = bb[0,0]
    x_origin = bb[0,1]
    
    y_end = bb[1,0]
    x_end = bb[1,1]

    tl = [y_origin+((y_end-y_origin)*0.2), x_origin+((x_end-x_origin)*0.2)]
    br = [y_end-((y_end-y_origin)*0.2), x_end-((x_end-x_origin)*0.2)]
    return np.array([tl, br])


def crop_image(im, bb, region):
    """
    returns a desired cropped region of the raw image

    im: raw image (numpy array)
    bb: the bounding box of the current region (defined by top left and bottom right corner points)
    region: 'TL', 'TR', 'BL', 'BR', 'centre'

    """
    
    if region == 'TL':
        new_bb = TL_bb(bb)
    elif region == 'TR':
        new_bb = TR_bb(bb)
    elif region == 'BL':
        new_bb = BL_bb(bb)
    elif region == 'BR':
        new_bb = BR_bb(bb)
    elif region == 'centre':
        new_bb = centre_bb(bb)


    y_start = new_bb[0,0]
    y_end = new_bb[1,0]
    x_start = new_bb[0,1]
    x_end = new_bb[1,1]
    im = im[int(y_start):int(y_end), int(x_start):int(x_end), :]
    return im, new_bb


test_image_ix = 3
bb1 = np.array([[0,0],list(raw_image_batch_list[test_image_ix].shape[:-1])])
t1 = raw_image_batch_list[test_image_ix]
t2, bb2 = crop_image(t1, bb1, 'centre')
t3, bb3 = crop_image(t1, bb2, 'centre')
t4, bb4 = crop_image(t1, bb3, 'centre')
t5, bb5 = crop_image(t1, bb4, 'centre')
t6, bb6 = crop_image(t1, bb5, 'centre')


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

T = 10
for t in range(T):

	ground_truth_bb_list = []
	current_bb_list = []
	for image_ix in range(len(raw_image_batch_list)):
		image = raw_image_batch_list[image_ix]
		image_name = img_name_list[image_ix]
		
		ground_truth_bb_list.append(image_actions.get_bb_gt(image_name))
		current_bb_list.append(np.array([[0,0],list(raw_image_batch_list[image_ix].shape[:-1])]))

		

