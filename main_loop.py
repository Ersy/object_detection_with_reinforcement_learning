import numpy as np
import argparse
import cv2

from keras.models import Sequential # part to build the mode
from keras.layers.core import Dense, Dropout, Activation, Flatten # types of layers and associated functions
from keras.optimizers import RMSprop, SGD, Adam #optimising method (cost function and update method)
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
batch_size_val = 5



# parser for the input, defining the number of training epochs and an image
parser = argparse.ArgumentParser(description = 'Epoch: ')
parser.add_argument('-n', metavar='N', type=int, default=0)
parser.add_argument("-i", "--image", help="path to the input image")
args = vars(parser.parse_args())
epochs_id = args['n']
image = args['image']

print(epochs_id)


### loading up VOC images of a given class
img_name_list = image_actions.get_img_names(VOC_path, 'aeroplane_trainval')
img_list = image_actions.load_images(VOC_path, img_name_list) 


### converting images of a given batch size to an image tensor 
image_batch_tensor = image_actions.batch_image_preprocessing(img_list[:batch_size_val]) #expects a list of PIL objects

""" 
### test code for classification

classifier_model = VGG16(include_top=True, weights='imagenet') #classifier VGG16 model
prediction = classifier_model.predict(image_batch_tensor, batch_size=batch_size_val)
P = imagenet_utils.decode_predictions(prediction)

for j in P:
	for (i, (imagenetID, label, prob)) in enumerate(j):
		print("{}. {}: {:.2f}%".format(i+1, label, prob*100))
	print("Next one!")

for im in image_batch_tensor:
	image_actions.view_image(im)

"""

### VGG16 model without top
vgg16_conv = VGG16(include_top=False, weights='imagenet')
conv_output = vgg16_conv.predict(image_batch_tensor, batch_size=batch_size_val)


### Q network definition

Q_network = Sequential()

num_of_actions = 4 #number of actions we choose from


"""
The input to the Q network should be the current state, i.e. the output of the VGG conv layers +
some additional information, e.g. the past actions
"""
Q_network.add(Flatten(input_shape=conv_output.shape[1:]))
Q_network.add(Dense(256, init='lecun_uniform')) #256 nodes, uniform initialisation, input 64 vec
Q_network.add(Activation('relu')) # relu activation

Q_network.add(Dense(256, init='lecun_uniform')) #256 nodes, uniform initialisation, input 64 vec
Q_network.add(Activation('relu')) # relu activation

Q_network.add(Dense(num_of_actions, init='lecun_uniform')) #164 nodes, uniform initialisation, input 64 vec
Q_network.add(Activation('linear')) # relu activation

sgd = SGD(lr=0.001, momentum=0.9)
rms = RMSprop(lr = 0.00001)
adam = Adam(lr=0.0001)
Q_network.compile(optimizer=adam, loss='mse')

target_test = np.array([[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,1,0,0],[0,0,1,0]])
#target_test = np.expand_dims(target_test, 0)
### training
learn_test = Q_network.fit(conv_output, target_test, batch_size=batch_size_val, nb_epoch=500, verbose=1)

print(Q_network.predict(conv_output, batch_size=batch_size_val))