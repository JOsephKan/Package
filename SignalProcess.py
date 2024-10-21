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

        Ck = np.zeros((ltime, int(llon/2)))
        Sk = np.zeros((ltime, int(llon/2)))

        for i in range(ltime):
            fourier_coeff = np.fft.fft(arr[i, :])

            Ck[i] = np.real(fourier_coeff[:int(llon/2)]*2)/llon
            Sk[i] = np.imag(fourier_coeff[:int(llon/2)]*2)/llon

        A = np.zeros((int(ltime/2), int(llon/2)))
        B = np.zeros((int(ltime/2), int(llon/2)))
        a = np.zeros((int(ltime/2), int(llon/2)))
        b = np.zeros((int(ltime/2), int(llon/2)))

        for i in range(int(llon/2)):
            Ck_coeff = np.fft.fft(Ck[:, i])
            Sk_coeff = np.fft.fft(Sk[:, i])

            A[:, i] = np.real(Ck_coeff[:int(ltime/2)]*2)/ltime
            B[:, i] = np.imag(Ck_coeff[:int(ltime/2)]*2)/ltime
            a[:, i] = np.real(Sk_coeff[:int(ltime/2)]*2)/ltime
            b[:, i] = np.imag(Sk_coeff[:int(ltime/2)]*2)/ltime

        return A, B, a, b
    
    def PowerSpectrum(self, arr):
        arr_fft = np.fft.fftshift(np.fft.fft2(arr))

        power_spec = (arr_fft * np.conj(arr_fft))[arr_fft.shape[0]//2:][:, ::-1]

        power_spec[1, :] *= 2

        power_spec /= np.prod(arr.shape)

        return power_spec.real

    # def CrossSpectrum(self, arr1, arr2):
    #     A1, B1, a1, b1 = self.FourierTransform(arr1)
    #     A2, B2, a2, b2 = self.FourierTransform(arr2)

    #     east_1 = 

    #     fft1 = np.fft.fftshift(np.fft.fft2(arr1))/np.prod(arr1.shape)
    #     fft2 = np.fft.fftshift(np.fft.fft2(arr2))/np.prod(arr2.shape)

    #     fft1_half = fft1[arr1.shape[0]//2:][:, ::-1]
    #     fft2_half = fft2[arr2.shape[0]//2:][:, ::-1]

    #     fft1_half[1:] *= 2
    #     fft2_half[1:] *= 2

    #     theta_1 = np.atan(fft1_half.imag / fft1_half.real)
    #     theta_2 = np.atan(fft2_half.imag / fft2_half.real)

    #     cross_spec = (fft2_half*fft1_half.conjugate())*np.cos(theta_1-theta_2)/2



    #     return cross_spec 
    # 
    def CrossSpectrum(self, arr1, arr2):
        A1, B1, a1, b1 = self.FourierTransform(arr1)
        A2, B2, a2, b2 = self.FourierTransform(arr2)

        east_1 = ((A1+b1) +1j * (B1-a1))/4
        east_2 = ((A2+b2) +1j * (B2-a2))/4
        west_1 = ((A1-b1) -1j * (B1+a1))/4
        west_2 = ((A2-b2) -1j * (B2+a2))/4

        fft_1 = np.concatenate([west_1[:, ::-1], east_1], axis=1)
        fft_2 = np.concatenate([west_2[:, ::-1], east_2], axis=1)

        theta_1 = np.arctan(fft_1.imag / fft_1.real)
        theta_2 = np.arctan(fft_2.imag / fft_2.real)

        cross_spec = 1/2 * (fft_1 * fft_2.conjugate()) * np.cos(theta_1-theta_2)

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
