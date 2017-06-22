import numpy as np
import argparse
import cv2

from keras.models import Sequential # part to build the mode
from keras.layers.core import Dense, Dropout, Activation, Flatten # types of layers and associated functions
from keras.optimizers import RMSprop, SGD #optimising method (cost function and update method)
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.applications import imagenet_utils
from keras.applications.vgg16 import preprocess_input, VGG16

### Local helpers
import image_actions

### 
from keras import backend as K
K.set_image_dim_ordering('tf')


### Vars
VOC_path = "/media/ersy/DATA/Google Drive/QM Work/Queen Mary/Course/Final Project/Reinforcement learning/VOCdevkit/VOC2007"
batch_size_val = 1



# parser for the input, defining the number of training epochs and an image
parser = argparse.ArgumentParser(description = 'Epoch: ')
parser.add_argument('-n', metavar='N', type=int, default=0)
parser.add_argument("-i", "--image", help="path to the input image")
args = vars(parser.parse_args())
epochs_id = args['n']
image = args['image']

print(epochs_id)


classifier_model = VGG16(include_top=True, weights='imagenet') #classifier VGG16 model


img_name_list = image_actions.get_img_names(VOC_path, 'aeroplane_trainval')
img_list = image_actions.load_images(VOC_path, img_name_list) 


# test with image batch
test_image = image_actions.batch_image_preprocessing(img_list[:batch_size_val]) #expects a list of PIL objects


prediction = classifier_model.predict(test_image, batch_size=batch_size_val)
P = imagenet_utils.decode_predictions(prediction)

for (i, (imagenetID, label, prob)) in enumerate(P[0]):
	print("{}. {}: {:.2f}%".format(i+1, label, prob*100))

for im in test_image:
	image_actions.view_image(im)

