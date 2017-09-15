from __future__ import division
from scipy.signal import convolve
import numpy as np


class LCS:
    """
    Class for local contour sequence extraction for opencv-style contours.
    See more in paper: #TODO
    """
    def __init__(self, n_points=-1, start_option=0, window_size=5, normalized=True, smooth=1):
        """
        Initialize LCS instance.
        :param n_points: number of points (`int`) to be used for feature vector extraction
        :param start_option: "top", "bottom", "left" or "right" - sets starting point to the extremal point of contour.
                              In case of `int`, sets starting point to be contour[idx]
        :param window_size: size (in points) of neighborhood for chord construction
        :param normalized: `boolean`, if vector should be normalized to it's max value
        :param smooth: size (`int`, odd) of a convolution kernel 
        """
        assert window_size % 2 == 1, "Window size must be odd"
        assert (n_points == -1) or (n_points > 1), "Invalid number of points"
        assert smooth % 2 == 1, "Smooth size must be odd"

        start_options = ["top", "left", "bottom", "right"]
        assert start_option in start_options or type(start_option) is int, \
            "Invalid start option, possible options are int, \"top\", \"left\", \"bottom\", \"right\""

        self.n_points = n_points
        self.start_option = start_option
        self.window_size = window_size
        self.normalized = normalized
        self.smooth = 1
        self.start_point_idx = 0

        if start_option == "top":
            self.compare = lambda p1, p2: p1.ravel()[1] < p2.ravel()[1]
        elif start_option == "left":
            self.compare = lambda p1, p2: p1.ravel()[0] < p2.ravel()[0]
        elif start_option == "bottom":
            self.compare = lambda p1, p2: p1.ravel()[1] > p2.ravel()[1]
        elif start_option == "right":
            self.compare = lambda p1, p2: p1.ravel()[0] > p2.ravel()[0]
        elif type(start_option) is int:
            self.compare = None
            self.start_point_idx = start_option
        else:
            self.compare = None

    @staticmethod
    def _calculate_distance_to_chord_(p, p0, p1):
        """
        Calculates euclidean distance from point `p` to chord between p0 and p1. 
        :param p: `np.array([x, y])`, point of interest
        :param p0: `np.array([x0, y0])`, chord starting point 
        :param p1: `np.array([x1, y1])`, chord ending point
        :return: distance, `np.double`
        """
        v = p1 - p0
        w = p - p0
        return np.abs(np.cross(v, w) / np.linalg.norm(v))

    def calculate(self, contour):
        """
        Calculates feature vector of LCS for a given `opencv-style` contour.
        :param contour: opencv-style contour (`np.array` with shape (n, 1, 2))
        :return: `np.array` of LCS, `np.array` of points used for LCS, `int` index of starting point in points
        """
        assert contour.size > 0, "Contour is empty"
        r = lambda i: i % contour.shape[0]

        if self.n_points == -1:
            n_points = contour.shape[0]
            step = 1
        else:
            n_points = self.n_points
            step = contour.shape[0] / self.n_points

        lcs = []
        points = []
        start_point_idx = self.start_point_idx
        for j in xrange(n_points):
            point = contour[np.int(j * step)]
            points.append(point)

            neighbors = contour[r(np.int(j * step) - self.window_size // 2), :], \
                        contour[r(np.int(j * step) + self.window_size // 2), :]

            lcs.append(self._calculate_distance_to_chord_(point, *neighbors))

            if self.compare is not None:
                if self.compare(point, contour[np.int(start_point_idx * step)]):
                    start_point_idx = j

        lcs = np.roll(np.array(lcs), - self.start_point_idx).ravel()

        if self.smooth > 1:
            kernel = np.array([1 / self.smooth] * self.smooth)
            lcs = convolve(lcs, kernel, mode='full')

        if self.normalized:
            lcs /= lcs.max()

        points = np.array(points)

        return lcs, points, start_point_idx
