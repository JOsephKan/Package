import numpy as np
from sklearn.decomposition import PCA

class Fourier:
    def __init__(self):
        self.A = None
        self.B = None
        self.a = None
        self.b = None

    def FourierTransform(self, arr):
        ltime, llon = arr.shape

        fourier_coeff = np.fft.fft(arr, axis=1)[:, :llon//2]*2/llon
        Ck = np.real(fourier_coeff)
        Sk = np.imag(fourier_coeff)

        Ck_coeff = np.fft.fft(Ck, axis=0)[:ltime//2]*2/ltime
        Sk_coeff = np.fft.fft(Sk, axis=0)[:ltime//2]*2/ltime

        A = np.real(Ck_coeff); B = np.imag(Ck_coeff)
        a = np.real(Sk_coeff); b = np.imag(Sk_coeff)

        return A, B, a, b
    
    def PowerSpectrum(self, arr):
        A, B, a, b = self.FourierTransform(arr)
        east = ((A+b)**2 + (B-a)**2)/8
        west = ((A-b)**2 + (-B-a)**2)/8

        arr_ps = np.concatenate([west[:, ::-1], east], axis=1)

        return arr_ps
   
    def CrossSpectrum(self, arr1, arr2):
        A1, B1, a1, b1 = self.FourierTransform(arr1)
        A2, B2, a2, b2 = self.FourierTransform(arr2)

        west1 = ((A1-b1)+1j*(-B1-a1))/4
        west2 = ((A2-b2)+1j*(-B2-a2))/4
        east1 = ((A1+b1)+1j*(B1-a1))/4
        east2 = ((A2+b2)+1j*(B2-a2))/4

        theta1 = np.arctan(np.divide(-B1 - a1, A1 - b1))
        theta2 = np.arctan(np.divide(-B2 - a2, A2 - b2))

        CO_west = np.multiply(west1, np.conj(west2)) * np.cos(theta2-theta1)/2
        CO_east = np.multiply(east1, np.conj(east2)) * np.cos(theta2-theta1)/2

        CO_total = np.concatenate([CO_west[:, ::-1], CO_east], axis=1)

        return CO_total


    def InverseFourierTransform(self, coeff_list):
        A, b, a, b = coeff_list
        ltime_h, llon_h = A.shape

        Ck = (A + 1j * B)*ltime_h
        Sk = (a + 1j * b)*ltime_h

        Ck = np.concatenate([Ck[::-1], Ck], axis=0)
        Sk = np.concatenate([Sk[::-1], Sk], axis=0)

        arr_fft = (np.fft.ifft(Ck, axis=0) + 1j * np.fft.ifft(Sk, axis=0))*llon_h

        arr_fft = np.concatenate([arr_fft[:, ::-1], arr_fft], axis=1)

        arr_recon = np.fft.ifft(arr_fft, axis=1).real

        return arr_recon

class EOF:
    """
    Calculating empirical orthogonal funcitons (EOFs)
    
    Parameters
    ----------
    dataset: tuple
        A tuple with elements are variables that you want to find their EOFs
        Variables must be array like, and must be standardized
        If given more than one dataset, combined EOF will be calculated
    
    n_components: int
        Number of modes that you need

    field: str, 1D or 2D, default = 2D
        The dimension of input variable arrays
    
    **svd_args: 
        Arguments for svd calculation in sklearn.decomposition.PCA, default = {"solver": "auto", "tol": 0.0, "iterated_power": "auto", "n_oversamples": 10, "power_iteration_normalizer": "auto", "random_state": None}
    
    About EOFs
    ----------
    The EOFs are vectors that represent the spatial distribution with largest temporal variation.
    In short, finding EOFs is equivalent to solving an eigenvalue problem of the variance matrix. The first eigen mode
    is EOF1, the second is EOF2, and so on.
    A variance matrix is done by multiplying the input variable array and its transpose, with temporal mean is zero.

    Note that
    ---------
    Original algorithm is developed by Kai-Chih Tseng: https://kuiper2000.github.io/
    """
    def __init__(
        self,
        dataset     : tuple,
        n_components: int,
        field       : str  = "2D",
        **svd_kwargs
    ):
        self.dataset      = dataset
        self.data_arr     = None
        self.n_components = n_components
        self.field        = field
        self.pca          = None
        self.EOF          = None
        self.PC           = None
        self.explained    = None
        self._svd         = svd_kwargs
    
    def _check_dimension(self):
        """
        If the dimensions of input variables are not consistent with self.field, raise ValueError

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for sub in self.dataset:
            if (self.field == "2D" and np.ndim(sub) == 3) or (self.field == "1D" and np.ndim(sub) == 2): pass
            else:
                raise ValueError("The dimensions of input variables need to be consistent with input 'field'")

    def _single_subdataset_reshape_2D(self, subdataset: np.ndarray) -> np.ndarray:
        """
        Reshape input array with dimension (time, space) into (time*space)

        Parameters
        ----------
        subdataset: array
            The array of variable with dimension (time, space)
        
        Returns
        -------
        _subdataset_new: array
            The array of variable reshaped to dimension (time*space)
        """
        _subdataset_new = np.reshape(subdataset, (subdataset.shape[0], subdataset.shape[1]*subdataset.shape[2]))
        return _subdataset_new

    def _dataset_reshape_2D(self) -> tuple:
        """
        if there are more than two variables:
            Transfer input tuple with variable arrays into np.ndarray,
            and reshape it from dimension (var, time, space1, space2) into (time, var*space1*space2)
            Assign self.data_arr as the reshaped array
        else:
            Reshape the variable array into (time, space1*space2)
            Assign self.data_arr as the reshaped array

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if len(self.dataset) > 1:
            arr           = np.array(self.dataset)
            self.data_arr = np.reshape(np.transpose(arr, (1, 0, 2, 3)), (arr.shape[1], arr.shape[0]*arr.shape[2]*arr.shape[3]))
        else:
            self.data_arr = self._single_subdataset_reshape_2D(self.dataset[0])
    
    def _dataset_reshape_1D(self):
        """
        Same as _dataset_reshape_2D, but for 1-dimensional input variables

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if len(self.dataset) > 1:
            arr           = np.array(self.dataset)
            self.data_arr = np.reshape(np.transpose(arr, (1, 0, 2)), (arr.shape[1], arr.shape[0]*arr.shape[2]))
        else:
            self.data_arr = self.dataset[0]

    def _fit(self):
        """
        Create a PCA class and fit it with input data

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pca_ = PCA(n_components = self.n_components, **self._svd)
        pca_.fit(self.data_arr)
        self.pca = pca_

    def _calc_EOF(self):
        """
        Calculate different EOF modes

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.EOF = self.pca.components_
    
    def _calc_PC(self):
        """
        Calculate PCs with input data and EOF modes

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        PC = np.dot(self.EOF, self.data_arr.T)
        self.PC = PC
    
    def _calc_explained(self):
        """
        Calculate the explainable ratio of each given EOF modes

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.explained = self.pca.explained_variance_ratio_

    def get(self):
        """
        Call _fit() _calc_EOF() _calc_PC _calc_explained() and calculate all of them

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._check_dimension()
        if self.field == "1D":
            self._dataset_reshape_1D()
        else:
            self._dataset_reshape_2D()
        self._fit()
        self._calc_EOF()
        self._calc_PC()
        self._calc_explained()

