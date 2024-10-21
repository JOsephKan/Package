import numpy as np
# def PowerSpectrum(data):
#      """
#      Compute the power spectrum of 2D space-time data (e.g., similar to Wheeler and Kiladis 1999).
#      
#      Args:
#          data (ndarray): 2D array representing space-time data (time, space).
#      
#      Returns:
#          power_spec (ndarray): The computed power spectrum (positive frequencies only).
#      """
#      # Step 1: Perform 2D Fourier Transform and shift the zero-frequency component to the center
#      data_fft = np.fft.fftshift(np.fft.fft2(data))
#      
#      # Step 2: Compute the power spectrum as the squared magnitude of the Fourier coefficients
#      power_spec_fft = np.abs(data_fft) ** 2
#      
#      # Step 3: Retain only positive frequencies (upper half of the spectrum)
#      positive_freq_spec = power_spec_fft[data.shape[0]//2:, :]  # Positive frequencies
#      
#      positive_freq_spec[1:] *= 2
#  
#      # Step 4: Normalize to ensure Parseval's identity holds
#      power_spec = positive_freq_spec / np.prod(data.shape)
#      
#      return power_spec

def PowerSpectrum(data):
    data_fft = np.fft.fftshift(np.fft.fft2(data))
    power_spec_fft = np.abs(data_fft)**2

    power_spec = power_spec_fft[data_fft.shape[0]//2:, :]

    power_spec[1:] *= 2

    power_spec /= np.prod(data.shape)

    return power_spec

def compute_power_spectrum_2d(data_2d):
    """
    Compute the power spectrum of a 2D array (space-time data) and verify Parseval's identity.
    Args:
        data_2d (ndarray): 2D array representing the space-time data.

    Returns:
        energy_space_time (float): Total energy in the space-time domain.
        energy_freq_wavenumber (float): Total energy in the frequency-wavenumber domain.
        power_spectrum (ndarray): 2D array representing the power spectrum.
    """

    # 1. Compute the total energy in the space-time domain
    energy_space_time = np.sum(np.abs(data_2d) ** 2)
  
    power_spectrum = PowerSpectrum(data_2d)

    # 4. Compute the total energy in the frequency-wavenumber domain (Parseval's theorem)
    energy_freq_wavenumber = np.sum(power_spectrum)

    # 5. Return the energies and power spectrum
    return energy_space_time, energy_freq_wavenumber, power_spectrum


# Generate the example 2D space-time data (sine wave pattern)
space = np.linspace(0, 2*np.pi, 100)  # spatial dimension
time = np.linspace(0, 2*np.pi, 100)   # temporal dimension
space_grid, time_grid = np.meshgrid(space, time)
data_2d = np.sin(space_grid) * np.cos(time_grid)  # Example space-time data

# Compute power spectrum and energies
energy_space_time, energy_freq_wavenumber, power_spectrum = compute_power_spectrum_2d(data_2d)

# Print results
print(f"Energy in space-time domain: {energy_space_time}")
print(f"Energy in frequency-wavenumber domain: {energy_freq_wavenumber}")

# Check if Parseval's identity holds
if np.isclose(energy_space_time, energy_freq_wavenumber, atol=1e-10):
    print("Parseval's identity holds.")
else:
    print("Parseval's identity does not hold.")
