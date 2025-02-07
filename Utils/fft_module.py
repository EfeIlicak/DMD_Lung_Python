# fft_module.py

import numpy as np

def fftc(arr, axis=-1):
    """
    Compute the centered 1D FFT of an array along the specified axis.

    Parameters:
    arr (ndarray): Input array.
    axis (int): Axis along which the FFT is computed. Default is the last axis.

    Returns:
    ndarray: Centered 1D FFT of the input array.
    """
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(arr, axes=axis), axis=axis), axes=axis)

def ifftc(arr, axis=-1):
    """
    Compute the centered 1D IFFT of an array along the specified axis.

    Parameters:
    arr (ndarray): Input array in the frequency domain.
    axis (int): Axis along which the IFFT is computed. Default is the last axis.

    Returns:
    ndarray: Centered 1D IFFT of the input array.
    """
    return np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(arr, axes=axis), axis=axis), axes=axis)


def fft2c(arr):
    """
    Compute the centered 2D FFT of a 2D array.

    Parameters:
    arr (ndarray): Input 2D array.

    Returns:
    ndarray: Centered 2D FFT of the input array.
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr)))

def ifft2c(arr):
    """
    Compute the centered 2D IFFT of a 2D array.

    Parameters:
    arr (ndarray): Input 2D array in the frequency domain.

    Returns:
    ndarray: Centered 2D IFFT of the input array.
    """
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(arr)))

def fft3c(arr):
    """
    Compute the centered 3D FFT of a 3D array.

    Parameters:
    arr (ndarray): Input 3D array.

    Returns:
    ndarray: Centered 3D FFT of the input array.
    """
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(arr)))

def ifft3c(arr):
    """
    Compute the centered 3D IFFT of a 3D array.

    Parameters:
    arr (ndarray): Input 3D array in the frequency domain.

    Returns:
    ndarray: Centered 3D IFFT of the input array.
    """
    return np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(arr)))
