import numpy as np

class gabor_gen():
    
    def __init__(self, im_size):
        self.im_size = im_size
        self.existing_gabor_loc = []
        
    def gen_image(self, num_of_gabors, gabor_size, sigma):
        """
        Generates an image of a select size with a select number of gabors
        """
        im_len = (np.linspace(0, self.im_size, self.im_size+1))
        x_mesh, y_mesh = np.meshgrid(im_len, im_len)
        im = x_mesh*0 + y_mesh*0
        bb = []
        
        for gab in range(num_of_gabors):
            gabor = self.gabor_patch(size=gabor_size, lambda_=5,theta=0, sigma=sigma, phase=0)
            x, y = self.gen_random_location(im_len, gabor_size)
            im[y:y+gabor_size,x:x+gabor_size] = gabor
            bb.append(np.array([[y, x],[y+gabor_size, x+gabor_size]]))
        return im, bb
    
    def gen_random_location(self, im_len, gabor_size):
        """
        Select a random location
        """
        x_choice = [x for x in im_len][:-gabor_size]
        y_choice = [y for y in im_len][:-gabor_size]
        x = int(np.random.choice(x_choice))
        y = int(np.random.choice(y_choice))

        return x, y
        
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
        gauss[gauss < trim] = 0

        return grating * gauss
