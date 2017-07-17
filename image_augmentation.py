# Image augmentation
# take in an image and the bounding box/es
# flip image horizontally
# determine new bounding box location based on this
import numpy as np

def flip_image(img, boundingbox):
	"""
	Takes an image and list of bounding boxes for the image 
	and flips everything horizontally
	returns the flipped image and boundingbox 
	(elements of the bb are changed inplace)
	"""
	flipped_image = np.fliplr(img)
	img_width = flipped_image.shape[1]
	for box_ix in range(len(boundingbox)):
		bb_topx = boundingbox[box_ix][0, 1]
		bb_bottomx = boundingbox[box_ix][1, 1]
		bb_width = bb_bottomx - bb_topx

		boundingbox[box_ix][0, 1] = img_width - bb_width - bb_topx	
		boundingbox[box_ix][1, 1] = img_width - bb_topx
	return flipped_image, boundingbox