import numpy as np
from scipy.fft import ifft2


def create_filtered_image(x_red_filt, x_green_filt, x_blue_filt, x, im_start, im_size):
    """
    Create the final filtered image by applying inverse Fourier transform.

    Args:
    x_red_filt, x_green_filt, x_blue_filt (np.array): Filtered color channels.
    x (np.array): Original image.
    im_start (int): Starting point for image cropping.
    im_size (int): Size of the image to process.

    Returns:
    np.array: Filtered image.
    """
    x_filt = x[im_start:im_size + im_start, :, :].copy()
    x_filt[:, :, 0] = np.abs(ifft2(x_red_filt))
    x_filt[:, :, 1] = np.abs(ifft2(x_green_filt))
    x_filt[:, :, 2] = np.abs(ifft2(x_blue_filt))
    return x_filt
