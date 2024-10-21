import numpy as np
from sklearn.decomposition import PCA

class Fourier:
    def __init__(self):
        pass

    def PowerSpectrum(self, arr):
        arr_fft = np.fft.fftshift(np.fft.fft2(arr))

        power_spec = (arr_fft * np.conj(arr_fft))[arr_fft.shape[0]//2:][:, ::-1]

        power_spec[1:, :] *= 2

        power_spec /= np.prod(arr.shape)

        return power_spec.real

    def CrossSpectrum(self, arr1, arr2):
        ano_1 = arr1 - arr1.mean()
        ano_2 = arr2 - arr2.mean()

        fft1 = np.fft.fftshift(np.fft.fft2(ano_1))[ano_1.shape[0]//2:][:, ::-1]
        fft2 = np.fft.fftshift(np.fft.fft2(ano_2))[ano_2.shape[0]//2:][:, ::-1]

        

        cross_spec = (fft1 * np.conj(fft2)) / (np.prod(fft1.shape))

        cross_spec[1:] *= 2

        return cross_spec

class EOF:
    def __init__(self, arr) -> None:
        self.arr = arr

    def NormalEquation(self, eof: np.ndarray) -> np.ndarray:
        xTx = np.linalg.inv(np.matmul(eof.T, eof))
        op = np.matmul(xTx, eof.T)
        normal = np.matmul(np.array(op), np.array(self.arr))

        return normal

    def EmpOrthFunc(self):
        CovMat = np.matmul(np.array(self.arr), np.array(self.arr.T)) / (self.arr.shape[1])

        eigvals, eigvecs = np.linalg.eig(CovMat)

        ExpVar = eigvals / eigvals.sum() 
        EOF = (eigvecs - eigvecs.mean()) / eigvecs.std()
        
        if EOF[:, 0][self.arr.shape[0]//2] < 0:
            EOF = -EOF

        else: 
            EOF = EOF

        PC = self.NormalEquation(EOF)

        return ExpVar, EOF, PC
