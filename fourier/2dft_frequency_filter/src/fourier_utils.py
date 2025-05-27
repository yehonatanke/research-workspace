import numpy as np
from scipy.fft import fft2, fftshift


def apply_fourier_transform(x_red, x_green, x_blue, img_ft):
    """
    Apply Fourier transform to the image channels.

    Args:
    x_red, x_green, x_blue (np.array): Image color channels.
    img_ft (int): Type of Fourier transform to apply.

    Returns:
    tuple: Transformed (x_red, x_green, x_blue)
    """
    if img_ft == 1:
        return fft2(x_red), fft2(x_green), fft2(x_blue)
    elif img_ft == 2:
        return np.abs(fft2(x_red)), np.abs(fft2(x_green)), np.abs(fft2(x_blue))
    elif img_ft == 3:
        return np.exp(1j * np.angle(fft2(x_red))), np.exp(1j * np.angle(fft2(x_green))), np.exp(1j * np.angle(fft2(x_blue)))
    else:
        raise ValueError(f"Invalid img_ft value: {img_ft}")


def create_filter(shape, filt_l, filt_domain, x_blue):
    """
    Create a filter based on the specified parameters.

    Args:
    shape (tuple): Shape of the filter.
    filt_l (float): Filter parameter.
    filt_domain (int): Type of filter to create.
    x_blue (np.array): Blue channel of the image.

    Returns:
    np.array: Created filter.
    """
    im_size = shape[0]
    if filt_domain == 1:
        filt = np.zeros(shape)
        filt[:filt_l, :filt_l] = np.ones((filt_l, filt_l)) / (filt_l ** 2)
        return fft2(filt)
    elif filt_domain in [2, 3]:
        filt_shift = np.ones(shape) if filt_domain == 3 else np.zeros(shape)
        filt_box_min, filt_box_max = im_size // 2 - filt_l, im_size // 2 + filt_l
        filt_shift[filt_box_min:filt_box_max + 1, filt_box_min:filt_box_max + 1] = 0 if filt_domain == 3 else 1
        return fftshift(filt_shift)
    elif filt_domain in [4, 5]:
        y, x = np.ogrid[-im_size // 2:im_size // 2, -im_size // 2:im_size // 2]
        mask = x * x + y * y <= filt_l * filt_l
        filt_shift = mask if filt_domain == 4 else ~mask
        return fftshift(filt_shift)
    elif filt_domain == 6:
        return fftshift(np.abs(fftshift(x_blue)) > (filt_l * 10 ** 2))
    else:
        raise ValueError(f"Invalid filt_domain value: {filt_domain}")


def apply_filter(x_red, x_green, x_blue, filt):
    """
    Apply the filter to each color channel.

    Args:
    x_red, x_green, x_blue (np.array): Fourier transformed color channels.
    filt (np.array): Filter to apply.

    Returns:
    tuple: Filtered (x_red, x_green, x_blue)
    """
    return x_red * filt, x_green * filt, x_blue * filt
