import numpy as np
def CrossSpectrum(arr1, arr2):
    fft1 = np.fft.fftshift(np.fft.fft2(arr1))
    fft2 = np.fft.fftshift(np.fft.fft2(arr2))

    cross_spec = (fft1 * np.conj(fft2))[arr1.shape[0]//2:][:, ::-1]

    cross_spec[1:] *= 2

    cross_spec = cross_spec / np.prod(arr1.shape)

    return cross_spec


def compute_cross_spectrum_2d(field1, field2):
    """
    Computes the cross-spectrum of two 2D fields and verifies Parseval's identity.
    
    Args:
        field1 (ndarray): First 2D field (e.g., space-time data).
        field2 (ndarray): Second 2D field (e.g., space-time data).
    
    Returns:
        cross_spectrum (ndarray): Cross-spectrum of field1 and field2 in the frequency-wavenumber domain.
        energy_space_time_domain (float): Total energy in the space-time domain.
        energy_freq_wavenumber_domain (float): Total energy in the frequency-wavenumber domain.
    """
    # 1. Compute the total energy in the space-time domain (sum of squares of both fields)
    energy_space_time_domain = np.sum(np.abs(field1) ** 2) + np.sum(np.abs(field2) ** 2)

    # 2. Perform 2D Fourier transforms of both fields
    fft_field1 = np.fft.fft2(field1)
    fft_field2 = np.fft.fft2(field2)

    # 3. Compute the cross-spectrum (fft_field1 * conjugate(fft_field2))
    cross_spectrum = fft_field1 * np.conj(fft_field2)

    # 4. Normalize to ensure Parseval's identity holds
    #cross_spectrum_normalized = cross_spectrum / np.prod(field1.shape)

    cross_spectrum_normalized = CrossSpectrum(field1, field2)

    # 5. Compute the total energy in the frequency-wavenumber domain
    energy_freq_wavenumber_domain = (np.sum(np.abs(fft_field1) ** 2) + np.sum(np.abs(fft_field2) ** 2)) / np.prod(field1.shape)

    return cross_spectrum_normalized, energy_space_time_domain, energy_freq_wavenumber_domain


# Example data (two 2D space-time fields)
field1 = np.random.randn(128, 128)  # Replace with actual data for field1
field2 = np.random.randn(128, 128)  # Replace with actual data for field2

# Compute the cross-spectrum and energies
cross_spectrum, energy_space_time, energy_freq_wavenumber = compute_cross_spectrum_2d(field1, field2)

# Print results to compare energies
print(f"Energy in space-time domain: {energy_space_time}")
print(f"Energy in frequency-wavenumber domain: {energy_freq_wavenumber}")

# Check if Parseval's identity holds
if np.isclose(energy_space_time, energy_freq_wavenumber, atol=1e-10):
    print("Parseval's identity holds.")
else:
    print("Parseval's identity does not hold.")
