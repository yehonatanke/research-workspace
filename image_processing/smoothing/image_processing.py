import numpy as np


def create_circle_image(size=80, radius=20):
    """Create a black image with a white circle."""
    image = np.zeros((size, size))
    y, x = np.ogrid[-size // 2:size // 2, -size // 2:size // 2]
    mask = x * x + y * y <= radius * radius
    image[mask] = 255
    return image


def create_square_image(size=80, side=40):
    """Create a black image with a white square."""
    image = np.zeros((size, size))
    start = (size - side) // 2
    end = start + side
    image[start:end, start:end] = 255
    return image


def create_triangle_image(size=80, side=40):
    """Create a black image with a white triangle."""
    image = np.zeros((size, size))
    height = int(side * np.sqrt(3) / 2)
    x = np.arange(size)
    y = np.arange(size)
    center_x, center_y = size // 2, size // 2
    mask = (2 * np.abs(x[None, :] - center_x) - np.abs(y[:, None] - center_y) <= side) & \
           (y[:, None] - center_y <= height // 2) & \
           (y[:, None] - center_y >= -height // 2)
    image[mask] = 255
    return image


def smooth_image(image):
    """
    Apply smoothing to the image using a 3x3 average filter.

    Implements the smoothing algorithm.
    It smooths the image by replacing each pixel's value with the average of its
    3x3 neighborhood.

    Logic:
    1. Create a copy of the input image to avoid modifying the original.
    2. Iterate over each pixel in the image (excluding the border pixels).
    3. For each pixel, calculate the average value of its 3x3 neighborhood.
    4. Replace the pixel's value in the smoothed image with this average.
    """
    smoothed = np.copy(image)
    rows, cols = image.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighborhood = image[i - 1:i + 2, j - 1:j + 2]
            smoothed[i, j] = np.mean(neighborhood)
    return smoothed
