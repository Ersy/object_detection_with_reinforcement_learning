from scipy.ndimage import zoom
import numpy as np
import random
import cPickle as pickle

class gabor_gen():
    
    def __init__(self, im_size):
        self.im_size = im_size
        self.existing_gabor_loc = []
        
    def gen_image(self, num_of_gabors, gabor_size, lambda_, theta,sigma, phase, noisy=False, beta=-2, random_scaling=False):
        """
        Generates an image of a select size with a select number of gabors
        """
        im_len = (np.linspace(0, self.im_size, self.im_size+1))
        x_mesh, y_mesh = np.meshgrid(im_len, im_len)
        bb = []
        
        if noisy:
            # create spatial noise background and normalise
            im = self.spatial_noise(beta)
            im = (im - im.mean())/im.std()
        else:
            im = x_mesh*0 + y_mesh*0
        
        for gab in range(num_of_gabors):
            scaling_factor = 1
            
            # create gabor and normalise
            gabor, gauss = self.gabor_patch(size=gabor_size, lambda_=lambda_,theta=theta,sigma=sigma, phase=phase)
            gabor = (gabor - gabor.mean())/gabor.std()
            
            # flag for random scaling of patches for variability
            if random_scaling:
                if bool(random.getrandbits(1)):
                    scaling_factor = 2
                    gabor = zoom(gabor, 2)
            
            scaled_gabor_size = gabor_size*scaling_factor
            
            x, y = self.gen_random_location(im_len, scaled_gabor_size)
            
            # reduce noise in the gabor region by 1-gaussian then add gabor patch
            im[y:y+scaled_gabor_size,x:x+scaled_gabor_size] = im[y:y+scaled_gabor_size,x:x+scaled_gabor_size]*(1-gauss)
            im[y:y+scaled_gabor_size,x:x+scaled_gabor_size] = im[y:y+scaled_gabor_size,x:x+scaled_gabor_size]+gabor

            
            bb.append(np.array([[y, x],[y+scaled_gabor_size, x+scaled_gabor_size]]))

        # 0-255 mapping
        im = self._convert_to_im(im)
            
        return im, bb
    
    def _convert_to_im(self, im):
        """
        converts image array values from original range to 0-255
        """
        input_min = im.min()
        input_max = im.max()
        output_min = 0
        output_max = 255
        
        input_range = input_max - input_min
        output_range = output_max - output_min

        new_im = ((im - input_min) * output_range / input_range) + output_min
        new_im = np.uint8(np.ceil(new_im))
        new_im = self.to_rgb1a(new_im)
        
        return new_im

    def to_rgb1a(self, im):
        """
        converts image from single channel to 3 channels
        code from: http://www.socouldanyone.com/2013/03/converting-grayscale-to-rgb-with-numpy.html (Matt Murfitt, 2013)
        """
        w, h = im.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  im
        return ret    
    
    
    def gen_random_location(self, im_len, gabor_size):
        """
        Selects a random location within the bounds of the image
        """
        x_choice = [x for x in im_len][:-gabor_size]
        y_choice = [y for y in im_len][:-gabor_size]
        x = int(np.random.choice(x_choice))
        y = int(np.random.choice(y_choice))

        return x, y
        
    def spatial_noise(self, beta):
        """
        generates a noisy background with a given power spectrum
        adapted from http://uk.mathworks.com/matlabcentral/fileexchange/5091-generate-spatial-data (Jon Yearsley, 2016)
        """
        DIM = [self.im_size,self.im_size]
        BETA = beta

        u1 = np.array(range(0,int(DIM[0]/2)+1, 1))
        u2 = -np.array(range(int(np.ceil(DIM[0]/2))-1, 0, -1))
        u = (np.hstack((u1, u2))/DIM[0])
        u = np.tile(u, (DIM[1],1)).T


        v1 = np.array(range(0,int(DIM[1]/2)+1, 1))
        v2 = -np.array(range(int(np.ceil(DIM[1]/2))-1, 0, -1))
        v = (np.hstack((v1, v2))/DIM[1])
        v = np.tile(v, (DIM[0],1))

        Spatial_freq = np.power(np.power(u, 2) + np.power(v, 2), (BETA/2.0))

        Spatial_freq[Spatial_freq == np.inf] =0

        phi = np.random.rand(DIM[0], DIM[1])

        a = np.power(Spatial_freq, 0.5)
        b = (np.cos(2*np.pi*phi))+(1j*np.sin(2*np.pi*phi))

        x = np.fft.ifft2(a*b)
        im = np.real(x)
        return im
        
        
    def gabor_patch(self, size, lambda_, theta, sigma, phase, trim=.005):
        """
        Create a Gabor Patch

        size : int
            Image size (n x n)

        lambda_ : int
            Spatial frequency (px per cycle)

        theta : int or float
            Grating orientation in degrees

        sigma : int or float
            gaussian standard deviation (in pixels)

        phase : float
            0 to 1 inclusive
        """
        # make linear ramp
        X0 = (np.linspace(1, size, size) / size) - .5

        # Set wavelength and phase
        freq = size / float(lambda_)
        phaseRad = phase * 2 * np.pi

        # Make 2D grating
        Xm, Ym = np.meshgrid(X0, X0)

        # Change orientation by adding Xm and Ym together in different proportions
        thetaRad = (theta / 360.) * 2 * np.pi
        Xt = Xm * np.cos(thetaRad)
        Yt = Ym * np.sin(thetaRad)
        grating = np.sin(((Xt + Yt) * freq * 2 * np.pi) + phaseRad)

        # 2D Gaussian distribution
        gauss =  np.exp(-((Xm ** 2) + (Ym ** 2)) / (2 * (sigma / float(size)) ** 2))
        
        # Trim
        cropped_gauss = gauss[gauss < trim] = 0

        return grating * gauss, gauss
        


import matplotlib.patches as patches
import matplotlib.pyplot as plt

gabor_size=30
sigma=5
num_of_pics = 500
num_of_gabors = 1
im_size = 224
beta = -2
noisy=True
phase=0
lambda_ = 6
theta=0

def generate_x_images(num_of_pics, im_size, num_of_gabors, gabor_size, lambda_, theta, phase, sigma, noisy):
    image_container = []
    bb_container = []
    gabor_instance = gabor_gen(im_size=im_size)
    for i in range(num_of_pics):
        image, bb = gabor_instance.gen_image(num_of_gabors=num_of_gabors, 
                                             gabor_size=gabor_size,
                                             lambda_ = lambda_,
                                             theta = theta,
                                             phase=phase,
                                             sigma=sigma, 
                                             beta=beta, 
                                             noisy=noisy)
        image_container.append(image)
        bb_container.append(bb)
    return image_container, bb_container

train_images, train_bbs = generate_x_images(num_of_pics, im_size, num_of_gabors, gabor_size, lambda_, theta, phase, sigma, noisy)

pickle.dump( train_images, open( "/media/ersy/Other/Google Drive/QM Work/Queen Mary/Course/Final Project/project_code/pickled_data/test_images_2_noisy.pickle", "wb" ) )
pickle.dump( train_bbs, open( "/media/ersy/Other/Google Drive/QM Work/Queen Mary/Course/Final Project/project_code/pickled_data/test_bbs_2_noisy.pickle", "wb" ) )
