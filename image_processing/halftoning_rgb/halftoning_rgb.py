import numpy as np
from PIL import Image


"""
Error Diffusion (Floyd-Steinberg):
In error diffusion, for each pixel, you approximate its color and compute the error between the original and the approximated color. The error is then diffused to the neighboring pixels in the image.

- Steps:
  - For each pixel, round its RGB values to either 0 or 255 (black or white).
  - Compute the error between the original pixel and the halftoned pixel.
  - Distribute the error to neighboring pixels (using the Floyd-Steinberg matrix) to ensure the total error in the image remains balanced.
  - This makes the halftoning less noticeable while preserving image detail.


 - Dithering Algorithm (Ordered Dithering)
Dithering uses a predefined pattern (dithering matrix) to modulate the color values of each pixel. The matrix is applied cyclically over the image, giving a patterned, halftone-like result.
  
- Steps:
  - A threshold matrix (like the Bayer matrix) is applied cyclically across the image.
  - For each pixel, the threshold from the matrix is compared to the pixel value.
  - If the pixel's value is greater than the threshold, it is set to white; otherwise, it is set to black.
  - This results in a patterned halftone effect.
"""

def floyd_steinberg_error_diffusion(image):
    """
    Applies Floyd-Steinberg error diffusion to an RGB image.
    
    Args:
    image: A PIL RGB image.
    
    Returns:
    A halftoned image.
    """
    image = np.array(image).astype(np.float32)  
    height, width, channels = image.shape
    
    # Define the error diffusion matrix (Floyd-Steinberg coefficients)
    diffusion_matrix = np.array([
        [0, 0, 7],
        [3, 5, 1]
    ]) / 16.0
    
    for y in range(height):
        for x in range(width):
            old_pixel = image[y, x].copy()  # Get current pixel (R, G, B)
            
            # Round the pixel value to either 0 or 255 for halftoning
            new_pixel = np.round(old_pixel / 255.0) * 255.0
            image[y, x] = new_pixel
            
            # Calculate the error as a vector for each channel
            error = old_pixel - new_pixel
            
            # Distribute the error to neighboring pixels using the diffusion matrix
            for dy in range(2):
                for dx in range(-1, 2):
                    if 0 <= y + dy < height and 0 <= x + dx < width:
                        image[y + dy, x + dx] += diffusion_matrix[dy, dx + 1] * error
    
    # Convert back to uint8 and return the image
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))

def ordered_dithering(image, dither_matrix=None):
    """
    Applies ordered dithering to an RGB image using a threshold matrix.
    
    Args:
    image: A PIL RGB image.
    dither_matrix: A dithering matrix (optional), default is a 4x4 Bayer matrix.
    
    Returns:
    A halftoned image.
    """
    if dither_matrix is None:
        # Use a default 4x4 Bayer matrix
        dither_matrix = np.array([
            [ 0,  8,  2, 10],
            [12,  4, 14,  6],
            [ 3, 11,  1,  9],
            [15,  7, 13,  5]
        ]) * (255 / 16)
    
    image = np.array(image).astype(np.float32)
    height, width, channels = image.shape
    dither_matrix_size = dither_matrix.shape[0]
    
    for y in range(height):
        for x in range(width):
            # For each channel, apply the dithering matrix
            for c in range(channels):
                threshold = dither_matrix[y % dither_matrix_size, x % dither_matrix_size]
                if image[y, x, c] > threshold:
                    image[y, x, c] = 255
                else:
                    image[y, x, c] = 0

    # Convert back to uint8 and return the image
    return Image.fromarray(image.astype(np.uint8))

def main():
  image_path = 'input/img1.png'
  image = Image.open(image_path).convert('RGB')
  
  # Apply Floyd-Steinberg error diffusion halftoning
  error_diffusion_image = floyd_steinberg_error_diffusion(image)
  error_diffusion_image.show()
  
  # Apply ordered dithering halftoning
  ordered_dithering_image = ordered_dithering(image)
  ordered_dithering_image.show()


if __name__ == "__main__":
    main()
