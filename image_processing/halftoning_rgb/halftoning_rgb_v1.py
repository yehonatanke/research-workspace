import numpy as np
from PIL import Image

"""
Error Diffusion algorithm:
a. For each pixel in the image:
   - Add the accumulated error to the current pixel value.
   - Compare the result to a threshold to determine if it should be white (255) or black (0).
   - Calculate the quantization error (difference between the old and new pixel values).
   - Distribute the error to the neighboring pixels that haven't been processed yet.
- We use the Floyd-Steinberg distribution pattern: 7/16 to the right, 3/16 to the bottom-left, 5/16 to the bottom, and 1/16 to the bottom-right.
- The error is treated as a vector (RGB), so each color channel is processed independently.


Dithering algorithm:
a. For each pixel in the image:
   - Calculate the corresponding threshold value from the Bayer matrix.
   - Compare the pixel value to the threshold to determine if it should be white (255) or black (0).
- We use a 4x4 Bayer matrix, which is tiled across the entire image.
- The matrix values are normalized to the range [0, 1] and then scaled to [0, 255] for comparison with pixel values.
"""

def error_diffusion(image, threshold=128):
    """
    Apply error diffusion halftoning to an RGB image.
    
    Args:
    image (numpy.ndarray): Input RGB image (height, width, 3)
    threshold (int): Threshold value for binarization (0-255)
    
    Returns:
    numpy.ndarray: Halftoned image
    """
    height, width, _ = image.shape
    output = np.zeros_like(image)
    error = np.zeros_like(image, dtype=float)
    
    for y in range(height):
        for x in range(width):
            old_pixel = image[y, x].astype(float) + error[y, x]
            new_pixel = np.where(old_pixel > threshold, 255, 0)
            output[y, x] = new_pixel
            
            quant_error = old_pixel - new_pixel
            
            if x + 1 < width:
                error[y, x + 1] += quant_error * 7/16
            if x - 1 >= 0 and y + 1 < height:
                error[y + 1, x - 1] += quant_error * 3/16
            if y + 1 < height:
                error[y + 1, x] += quant_error * 5/16
            if x + 1 < width and y + 1 < height:
                error[y + 1, x + 1] += quant_error * 1/16
    
    return output.astype(np.uint8)

def ordered_dithering(image):
    """
    Apply ordered dithering halftoning to an RGB image.
    
    Args:
    image (numpy.ndarray): Input RGB image (height, width, 3)
    
    Returns:
    numpy.ndarray: Halftoned image
    """
    height, width, _ = image.shape
    output = np.zeros_like(image)
    
    # Bayer matrix 4x4
    bayer_matrix = np.array([
        [ 0, 8, 2, 10],
        [12, 4, 14, 6],
        [ 3, 11, 1, 9],
        [15, 7, 13, 5]
    ]) / 16
    
    for y in range(height):
        for x in range(width):
            threshold = bayer_matrix[y % 4, x % 4] * 255
            output[y, x] = np.where(image[y, x] > threshold, 255, 0)
    
    return output.astype(np.uint8)

def main():
    input_image = np.array(Image.open("input/image1.png"))
    
    # Apply error diffusion
    error_diffusion_result = error_diffusion(input_image)
    Image.fromarray(error_diffusion_result).save("error_diffusion_result.png")
    
    # Apply dithering
    dithering_result = ordered_dithering(input_image)
    Image.fromarray(dithering_result).save("output/dithering_result.png")


if __name__ == "__main__":
    main()

