# This pacakge is for processing data
import numpy as np

class Format:
    def __init__(self, lat: np.ndarray):
        self.lat = lat
        self.latr = np.cos(np.deg2rad(lat))[None, :, None]
        self.lats = np.sum(self.latr)

    def sym(self, arr: np.ndarray):
        """
        Calculate the symmetrical array based on the input latitude and array.

        Args:
            lat (np.ndarray): The latitude array.
            arr (np.ndarray): The input array.

        Returns:
            np.ndarray: The symmetrical array.

        Input shape: (time, lat, lon)
        """
        sym_arr = np.sum (arr * self.latr, axis=1) / self.lats

        return sym_arr
    
    def asy(self, data: np.ndarray):
        """
        Calculate the asymmetric component of the given data based on latitude.

        Parameters:
            lat (np.ndarray): Array of latitudes.
            data (np.ndarray): Array of data.

        Returns:
            np.ndarray: Array containing the asymmetric component of the data.

        """
        idx = np.where(self.lat < 0)

        data_asy = data * self.latr

        data_asy[idx] *= -1

        data_asy = np.sum(data_asy, axis=1) / self.lats

        return data_asy

def GaussianFilter(arr: np.ndarray, num_of_pass:np.int64):
    arr_bg = arr.copy()

    kernel = np.array([1, 2, 1]) / 4.0

    for _ in range(num_of_pass):
        for i in range(arr.shape[0]):
            arr_bg[i] = np.convolve(arr_bg[i], kernel, mode="same")

    return arr_bg

