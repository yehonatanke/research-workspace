import cv2
import numpy as np


def load_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to load the image. Make sure '{file_path}' is in the current directory.")
    return image


def local_histogram_equalization(image, kernel_size=3):
    pad = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            local_region = padded_image[i:i + kernel_size, j:j + kernel_size]

            local_hist, _ = np.histogram(local_region.flatten(), 256, [0, 256])
            local_cdf = local_hist.cumsum()
            local_cdf_normalized = (local_cdf - local_cdf.min()) * 255 / (local_cdf.max() - local_cdf.min())

            output[i, j] = local_cdf_normalized[image[i, j]]

    return output.astype(np.uint8)
