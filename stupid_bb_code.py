import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = "/media/ersy/DATA/Google Drive/QM Work/Queen Mary/Course/Final Project/Reinforcement learning/VOCdevkit/VOC2007/JPEGImages/000007.jpg"

im = Image.open(image)

im_array = np.array(im)

plt.imshow(im_array)


bb1 = np.array([[0,0],list(im_array.shape[:-1])])

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


t1 = im_array
t2, bb2 = crop_image(im_array, bb1, 'TL')
t3, bb3 = crop_image(im_array, bb2, 'TR')
t4, bb4 = crop_image(im_array, bb3, 'BL')
t5, bb5 = crop_image(im_array, bb4, 'BR')
t6, bb6 = crop_image(im_array, bb5, 'centre')

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




