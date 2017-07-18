import image_actions

def get_class_images(VOC_path, desired_class, img_name_list, img_list):
	desired_class = image_actions.class_name_dict[desired_class]

	### loading up VOC images of a given class
	# img_name_list = image_actions.get_img_names(VOC_path, 'aeroplane_trainval')
	# img_list = image_actions.load_images(VOC_path, img_name_list) 


	desired_class_list_bb = []
	desired_class_list_image = []
	desired_class_list_name = []

	# collect bounding boxes for each image
	for image_ix in range(len(img_name_list)):
		current_image_groundtruth = []
		ground_image_bb_gt = image_actions.get_bb_gt(VOC_path, img_name_list[image_ix])
		image_flag = False	
		for ix in range(len(ground_image_bb_gt[0])):	
			if ground_image_bb_gt[0][ix] == desired_class:
				current_image_groundtruth.append(ground_image_bb_gt[1][ix])
				image_flag = True	
		if image_flag:
			desired_class_list_bb.append(current_image_groundtruth)	
			desired_class_list_image.append(img_list[image_ix])
			desired_class_list_name.append(img_name_list[image_ix])

	return desired_class_list_image, desired_class_list_bb, desired_class_list_name
