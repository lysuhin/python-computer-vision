import numpy as np


class SimpleBackgrounder:
    """
    Class for simple background subtraction. 
    Background is calculated as mean of first `train_length` images, then it's subtracted from every
    new image, and the resulting difference is thresholded using `threshold` value.
    """
    def __init__(self, background=None, threshold=127, train_length=50, dark_background=True):
        """
        Initialize SimpleBackgrounder class.
        :param background: image to be used as a background (ignoring following 2 parameters) 
        :param threshold: value to be uses as threshold for mask
        :param train_length: number of images to be used for mean calculation of background
        :param dark_background: flag to show if background is expected to be darker than objects
        """
        self.background = background
        self.backgrounds = []
        self.threshold = threshold
        self.dark_background = dark_background
        self.train_length = train_length

    @staticmethod
    def apply_mask(image, mask):
        """
        Apply mask to image.
        :param image: source image
        :param mask: mask image
        :return: masked version of image
        """
        result = image.copy()
        result[mask == 0] = 0
        return result

    def subtract(self, image):
        """
        Subtract background from image with correct numeric values.
        :param image: source image
        :return: difference between image and background
        """
        if self.dark_background:
            difference = np.subtract(image.astype(np.int), self.background.astype(np.int))
        else:
            difference = - np.subtract(image.astype(np.int), self.background.astype(np.int))
        difference[difference < 0] = 0
        difference[difference > 255] = 255
        return np.uint8(difference)

    def _threshold(self, image):
        """
        Threshold image.
        :param image: source image
        :return: thresholded image
        """
        return np.uint(image > self.threshold)

    def get_mask(self, image):
        """
        Calculate mask using background and threshold class parameters.
        :param image: source image
        :return: binary mask of 0's and 1's
        """
        mask = self._threshold(self.subtract(image))
        return mask

    def apply(self, image):
        """
        Complete processing of an image.
        :param image: source image
        :return: masked image
        """
        if self.background is None:
            if len(self.backgrounds) < self.train_length:
                self.backgrounds.append(image)
                return image
            elif len(self.backgrounds) == self.train_length:
                self.background = np.mean(np.array(self.backgrounds), axis=0).astype(np.uint8)

        mask = self.get_mask(image)
        result = self.apply_mask(image, mask)
        return result


class GaussianBackgrounder:
    """
    Class for background subtraction using per-pixel intensity distribution over train set of backgrounds.
    `Background limits` are calculated as mean +- sigma*std of first `train_length` images. Every pixel on
    new image is considered as `background` iff it's value lies between limits.
    """
    def __init__(self, background=None, train_length=50, sigma=3, dark_background=True):
        """
        Initialize GaussianBackgrounder class.
        :param background: image to be used as a background (ignoring following 2 parameters)
        :param train_length: number of images to be used for background modelling
        :param sigma: value that shows the background intensity bandwidth. Used `as is` if one background 
        image is passed and is used as a number of stds in case of using train set.
        :param dark_background: flag to show if background is expected to be darker than objects
        """
        self.backgrounds = []
        self.background = background
        self.sigma = sigma

        if self.background is not None:
            mean = self.background.mean(axis=(0, 1))
            self.low = self.background - self.sigma
            self.high = self.background + self.sigma
        else:
            self.low = None
            self.high = None

        self.dark_background = dark_background
        self.train_length = train_length

    @staticmethod
    def apply_mask(image, mask):
        """
        Apply mask to image.
        :param image: source image
        :param mask: mask image
        :return: masked version of image
        """
        result = image.copy()
        result[mask == 0] = 0
        return result

    def get_mask(self, image):
        """
        Calculate mask using background and threshold class parameters.
        :param image: source image
        :return: binary mask of 0's and 1's
        """
        mask = np.zeros_like(image, dtype=np.uint) + 1
        mask[np.logical_and(image >= self.low, image <= self.high)] = 0
        return mask

    def apply(self, image):
        """
        Complete processing of an image.
        :param image: source image
        :return: masked image
        """
        if self.background is None:
            if len(self.backgrounds) < self.train_length:
                self.backgrounds.append(image)
                return image
            elif len(self.backgrounds) == self.train_length:
                self.background = np.mean(np.array(self.backgrounds), axis=0).astype(np.uint8)
                std = np.std(np.array(self.backgrounds), axis=0).astype(np.uint8)
                self.low = self.background - self.sigma * std
                self.high = self.background + self.sigma * std

        mask = self.get_mask(image)
        result = self.apply_mask(image, mask)
        return result

