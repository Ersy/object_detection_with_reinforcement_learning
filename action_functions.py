import numpy as np

# dictionary mapping Q output index to actions
action_dict = {0:'right',1:'down',2:'left',3:'up'}

# amount to update the corner positions by for each step
update_step = 0.1

def TL_right(bb):
	"""moves the top corner to the right"""
	y_origin = bb[0,0]
	x_origin = bb[0,1]
	
	y_end = bb[1,0]
	x_end = bb[1,1]

	pixel_update = int((x_end - x_origin) * update_step)

	x_origin = x_origin + pixel_update

	tl = [y_origin, x_origin]
	br = [y_end, x_end]
	return np.array([tl, br])


def TL_down(bb):
	"""moves the top corner to the right"""
	y_origin = bb[0,0]
	x_origin = bb[0,1]
	
	y_end = bb[1,0]
	x_end = bb[1,1]

	pixel_update = int((y_end - y_origin) * update_step)

	y_origin = y_origin + pixel_update

	tl = [y_origin, x_origin]
	br = [y_end, x_end]
	return np.array([tl, br])


def BR_left(bb):
	"""moves the bottom corner to the left"""
	y_origin = bb[0,0]
	x_origin = bb[0,1]
	
	y_end = bb[1,0]
	x_end = bb[1,1]

	pixel_update = int((x_end - x_origin) * update_step)

	x_end = x_end - pixel_update

	tl = [y_origin, x_origin]
	br = [y_end, x_end]
	return np.array([tl, br])


def BR_up(bb):
	"""moves the top corner to the right"""
	y_origin = bb[0,0]
	x_origin = bb[0,1]
	
	y_end = bb[1,0]
	x_end = bb[1,1]

	pixel_update = int((y_end - y_origin) * update_step)

	y_end = y_end - pixel_update

	tl = [y_origin, x_origin]
	br = [y_end, x_end]
	return np.array([tl, br])


def crop_image(im, bb_in, region):
	"""
	returns a desired cropped region of the raw image

	im: raw image (numpy array)
	bb: the bounding box of the current region (defined by top left and bottom right corner points)
	region: 'TL', 'TR', 'BL', 'BR', 'centre'

	"""

	if action_dict[region] == 'right':
		new_bb = TL_right(bb_in)
	elif action_dict[region] == 'down':
		new_bb = TL_down(bb_in)
	elif action_dict[region] == 'left':
		new_bb = BR_left(bb_in)
	elif action_dict[region] == 'up':
		new_bb = BR_up(bb_in)

	y_start = new_bb[0,0]
	y_end = new_bb[1,0]
	x_start = new_bb[0,1]
	x_end = new_bb[1,1]

	# crop image to new boundingbox extents
	im = im[int(y_start):int(y_end), int(x_start):int(x_end), :]
	return im, new_bb

