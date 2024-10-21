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

        data_asy[idx] = -data_asy[idx]

        data_asy = np.sum(data_asy, axis=1) / self.lats

        return data_asy

def GaussianFilter(arr: np.ndarray, num_of_pass:np.int64):
    arr_bg = arr.copy()

    for _ in range(num_of_pass):
        print(arr_bg.shape)
        print(np.array([arr_bg[:, 1]]).shape)
        left = np.concatenate((np.array([arr_bg[:, 1]]).T, arr_bg[:, :-1]), axis=1)
        right = np.concatenate((arr_bg[:, 1:], np.array([arr_bg[:, -2]]).T), axis=1)

        arr_bg = (2*arr_bg+left+right)/4

    return arr_bg
