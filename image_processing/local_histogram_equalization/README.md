# Local Histogram Equalization 

Implements local histogram equalization for image enhancement.

## Files Description

- `input/embedded_squares.JPG`: The original input image to be processed.
- `output/embedded_squares.png`: The enhanced image after local histogram equalization.
- `src/image_processing.py`: Contains the local histogram equalization algorithm.
- `src/main.py`: The main script to run the image processing pipeline.
- `src/visualization.py`: Handles the visualization of images and histograms.


## Input/Output Example

### Input
The input image `embedded_squares.JPG` should be a grayscale image containing embedded squares with varying intensities.

![embedded_squares.JPG](input%2Fembedded_squares.JPG)

### Output
The output image `embedded_squares.png` will show enhanced local contrast, making the embedded squares more visible.

![embedded_squares.png](output%2Fembedded_squares.png)

## Algorithm Overview

The local histogram equalization algorithm works as follows:

1. For each pixel in the image:
    - Extract a local neighborhood around the pixel.
    - Compute the histogram of this local region.
    - Apply histogram equalization to this local histogram.
    - Use the equalized histogram to determine the new value of the center pixel.

2. This process enhances local contrast, making details more visible in different parts of the image.
