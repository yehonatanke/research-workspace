# Laplacian Filter Image Sharpening

## Overview

Implements Laplacian filter for image sharpening using frequency domain processing. It applies a Laplacian filter to grayscale images in the frequency domain, resulting in sharpened images with enhanced edges and details. Processes multiple images and provides visualization of intermediate steps.

## Laplacian Filter in Frequency Domain

The Laplacian filter is a second-order derivative filter used for edge detection and image sharpening. In the frequency domain, it emphasizes high-frequency components, which correspond to edges and fine details in the image.

### How the Laplacian Filter Works

1. Convert the image to the frequency domain using the Fourier Transform.
2. Apply the Laplacian filter in the frequency domain.
3. Convert the filtered image back to the spatial domain using the Inverse Fourier Transform.
4. Combine the filtered image with the original to enhance edges and details.

## Formal Description

The Laplacian filter in the frequency domain is based on the properties of the Fourier Transform and the Laplacian operator. Here's a detailed explanation of the mathematical principles behind this program:

### 1. Fourier Transform

The 2D Fourier Transform $F(u,v)$ of an image $f(x,y)$ is given by:

$$ F(u,v) = \int\int f(x,y) \cdot e^{-j2\pi(ux+vy)} \, dx \, dy $$

where $u$ and $v$ are the frequency variables corresponding to $x$ and $y$ in the spatial domain.

### 2. Laplacian Operator

The Laplacian operator $\nabla^2$ in the spatial domain is defined as:

$$ \nabla^2f = \frac{\partial^2f}{\partial x^2} + \frac{\partial^2f}{\partial y^2} $$

### 3. Laplacian in Frequency Domain

The Fourier Transform of the Laplacian operator is:

$$ \mathcal{F}\{\nabla^2f\} = -4\pi^2(u^2 + v^2)F(u,v) $$

This means that applying the Laplacian in the spatial domain is equivalent to multiplying the Fourier Transform of the image by $-4\pi^2(u^2 + v^2)$ in the frequency domain.

### 4. Implementing the Filter

In the program, we create the Laplacian filter $H(u,v)$ in the frequency domain as:

$$ H(u,v) = -4\pi^2((u - M/2)^2 + (v - N/2)^2) $$

where $M$ and $N$ are the dimensions of the image. The terms $(u - M/2)$ and $(v - N/2)$ are used to center the filter.

### 5. Applying the Filter

The filtered image $G(u,v)$ in the frequency domain is obtained by multiplying $F(u,v)$ with $H(u,v)$:

$$ G(u,v) = H(u,v) \cdot F(u,v) $$

### 6. Image Enhancement

The final enhanced image $g(x,y)$ is obtained by combining the original image $f(x,y)$ with the inverse Fourier Transform of $G(u,v)$:

$$ g(x,y) = f(x,y) + c \cdot \mathcal{F}^{-1}\{G(u,v)\} $$

where $c$ is a scaling factor (usually negative) that controls the degree of enhancement.

### 7. Effect on Image

- Low frequencies ($(u^2 + v^2)$ small): $H(u,v)$ is small, so these components are minimally affected.
- High frequencies ($(u^2 + v^2)$ large): $H(u,v)$ is large, so these components are amplified.

This results in an emphasis on edges and fine details in the image, which appear as high-frequency components in the frequency domain.

By applying the Laplacian filter in the frequency domain, we can achieve edge enhancement and sharpening across the entire image simultaneously, which can be more efficient than applying spatial domain filters, especially for large images.

### Implementation in the Code

The main Laplacian filter logic is implemented in several functions in `src/laplacian_frequency_filter.py`:

```python
def fourier_transform(image):
    # Compute 2D Fourier Transform
    ft = np.fft.fft2(image)
    # Shift the zero-frequency component to the center of the spectrum
    return np.fft.fftshift(ft)

def create_laplacian_filter(shape):
    P, Q = shape
    H = np.zeros((P, Q), dtype=np.float32)
    for u in range(P):
        for v in range(Q):
            # Compute the Laplacian filter value for each frequency
            H[u,v] = -4 * np.pi**2 * ((u - P/2)**2 + (v - Q/2)**2)
    return H

def apply_laplacian_filter(ft_image, laplacian_filter):
    # Apply the filter in the frequency domain
    Lap = laplacian_filter * ft_image
    # Shift back the zero-frequency component
    Lap = np.fft.ifftshift(Lap)
    # Perform inverse Fourier Transform and take the real part
    return np.real(np.fft.ifft2(Lap))
```

## Input-Output Examples

The program processes images placed in the `input` directory. For each image, it generates several outputs in the `output` directory:

1. Original Image
2. Fourier Transform Magnitude
3. Laplacian Filter
4. Laplacian-Filtered Image
5. Enhanced Image

### Example: Processing "blurry-moon.tif"

Input: input/blurry-moon.tif (original grayscale image)

Outputs:
- output/blurry-moon_Original_Image.png
- output/blurry-moon_Fourier_Transform_Magnitude.png
- output/blurry-moon_Laplacian_Filter.png
- output/blurry-moon_Laplacian_Filtered_Image.png
- output/blurry-moon_Enhanced_Image.png

As the Laplacian filter is applied, you'll see the emphasis on edges and fine details in the image. The final enhanced image will show improved sharpness and detail compared to the original.

## Functions

### In `src/laplacian_frequency_filter.py`:

- `load_and_normalize_image(image_path)`: Loads and normalizes an image to the range [0, 1].
- `fourier_transform(image)`: Computes the 2D Fourier Transform of the image.
- `create_laplacian_filter(shape)`: Creates a Laplacian filter in the frequency domain.
- `apply_laplacian_filter(ft_image, laplacian_filter)`: Applies the Laplacian filter to the Fourier Transform of the image.
- `enhance_image(original, laplacian)`: Combines the original image with the Laplacian-filtered image for enhancement.
- `laplacian_filter_workflow(image_path, save_to_dir)`: Orchestrates the entire Laplacian filtering process for a single image.

### In `src/main.py`:

- `process_images(image_directory, save_output)`: Processes all images in the specified directory using the Laplacian filter workflow.

## Features

- Laplacian filtering in the frequency domain
- Visualization of intermediate steps (Fourier Transform, Laplacian Filter)
- Side-by-side display of original and enhanced images
- Automatic saving of all intermediate and final outputs
- Batch processing of multiple images

## Examples

Here are some examples of the Laplacian filter process applied to different images.

### 1. Blurry Moon

#### Original Image:
![Blurry Moon Original](input/blurry-moon.tif)

#### Enhanced Image:
![Blurry Moon Enhanced](output/blurry-moon_Enhanced_Image.png)

### 2. Chronometer

#### Original Image:
![Chronometer Original](input/Chronometer.tif)

#### Enhanced Image:
![Chronometer Enhanced](output/Chronometer_Enhanced_Image.png)

### 3. Lunar Shadows

#### Original Image:
![Lunar Shadows Original](input/lunarshadows.tif)

#### Enhanced Image:
![Lunar Shadows Enhanced](output/lunarshadows_Enhanced_Image.png)

### 4. Magnified Pollen (Dark)

#### Original Image:
![Magnified Pollen Original](input/magnified-pollen-dark.tif)

#### Enhanced Image:
![Magnified Pollen Enhanced](output/magnified-pollen-dark_Enhanced_Image.png)

As you can see from these examples:

1. The Laplacian filter enhances edges and fine details in the images.
2. The enhanced images show improved contrast and sharpness compared to the originals.
3. The technique is particularly effective at bringing out textures and subtle features in the images.
4. It works well on various types of images, from astronomical objects to microscopic structures.

