import numpy as np

# dictionary mapping Q output index to actions
action_dict = {0:'TL',1:'TR',2:'BR',3:'BL',4:'centre'}


def TL_bb(bb):
	"""Takes a bounding box and returns a bounding box of the top left region"""
	y_origin = bb[0,0]
	x_origin = bb[0,1]
	
	y_end = bb[1,0]
	x_end = bb[1,1]

	tl = [y_origin, x_origin]
	br = [((y_end-y_origin)*0.6)+y_origin, ((x_end-x_origin)*0.6)+x_origin]
	return np.array([tl, br])


def TR_bb(bb):
	"""Takes a bounding box and returns a bounding box of the top right region"""
	y_origin = bb[0,0]
	x_origin = bb[0,1]
	
	y_end = bb[1,0]
	x_end = bb[1,1]

	tl = [y_origin, x_origin+((x_end-x_origin)*0.4)]
	br = [((y_end-y_origin)*0.6)+y_origin, x_end]
	return np.array([tl, br])


def BL_bb(bb):
	"""Takes a bounding box and returns a bounding box of the bottom left region"""
	y_origin = bb[0,0]
	x_origin = bb[0,1]
	
	y_end = bb[1,0]
	x_end = bb[1,1]

	tl = [y_origin+((y_end-y_origin)*0.4), x_origin]
	br = [y_end, ((x_end-x_origin)*0.6)+x_origin]
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


def crop_image(im, bb_in, region):
	"""
	returns a desired cropped region of the raw image

	im: raw image (numpy array)
	bb: the bounding box of the current region (defined by top left and bottom right corner points)
	region: 'TL', 'TR', 'BL', 'BR', 'centre'

	"""
	
	if action_dict[region] == 'TL':
		new_bb = TL_bb(bb_in)
	elif action_dict[region] == 'TR':
		new_bb = TR_bb(bb_in)
	elif action_dict[region] == 'BL':
		new_bb = BL_bb(bb_in)
	elif action_dict[region] == 'BR':
		new_bb = BR_bb(bb_in)
	elif action_dict[region] == 'centre':
		new_bb = centre_bb(bb_in)

	y_start = new_bb[0,0]
	y_end = new_bb[1,0]
	x_start = new_bb[0,1]
	x_end = new_bb[1,1]

	im = im[int(y_start):int(y_end), int(x_start):int(x_end), :]
	return im, new_bb

