import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_and_normalize_image(image_path):
    """
    Load an image using PIL and normalize its values to the range [0, 1].

    Args:
    image_path (str): Path to the image file.

    Returns:
    numpy.ndarray: Normalized image array.
    """
    # Open the image and convert to grayscale
    with Image.open(image_path) as img:
        f = np.array(img.convert('L')).astype(float)
    # Normalize pixel values to the range [0, 1]
    return f / 255


def save_image(image, title, image_path, save_to_dir=False):
    """
    Save an image to a specified directory.

    Args:
    image (numpy.ndarray): Image array to save.
    title (str): Title for the image file.
    image_path (str): Original image path (used to extract the image name).
    save_to_dir (bool): Whether to save the image or not.
    """
    if save_to_dir:
        # Extract the image name from the path
        image_name = image_path.split('/')[-1].split('.')[0]
        # Create the output path with the image name
        output_directory = 'output'
        output_path = f'{output_directory}/{image_name}_{title}.png'

        # Ensure the image is in the correct format for saving
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        Image.fromarray(image).save(output_path)


def display_image(image, title, image_path, save_to_dir=False):
    """
    Display an image using matplotlib and optionally save it.

    Args:
    image (numpy.ndarray): Image array to display.
    title (str): Title for the plot.
    image_path (str): Original image path (used for saving).
    save_to_dir (bool): Whether to save the image or not.
    """
    plt.figure(dpi=150)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.show()

    save_image(image, title, image_path, save_to_dir)


def fourier_transform(image):
    """
    Perform 2D Fourier Transform on the input image.

    Args:
    image (numpy.ndarray): Input image array.

    Returns:
    numpy.ndarray: Fourier Transform of the image (shifted to center).
    """
    # Compute 2D Fourier Transform
    ft = np.fft.fft2(image)
    # Shift the zero-frequency component to the center of the spectrum
    return np.fft.fftshift(ft)


def create_laplacian_filter(shape):
    """
    Create a Laplacian filter in the frequency domain.

    Args:
    shape (tuple): Shape of the filter (should match the image shape).

    Returns:
    numpy.ndarray: Laplacian filter in the frequency domain.
    """
    P, Q = shape
    H = np.zeros((P, Q), dtype=np.float32)
    for u in range(P):
        for v in range(Q):
            # Compute the Laplacian filter value for each frequency
            H[u, v] = -4 * np.pi ** 2 * ((u - P / 2) ** 2 + (v - Q / 2) ** 2)
    return H


def apply_laplacian_filter(ft_image, laplacian_filter):
    """
    Apply the Laplacian filter to the Fourier Transform of the image.

    Args:
    ft_image (numpy.ndarray): Fourier Transform of the image.
    laplacian_filter (numpy.ndarray): Laplacian filter in the frequency domain.

    Returns:
    numpy.ndarray: Laplacian-filtered image in the spatial domain.
    """
    # Apply the filter in the frequency domain
    Lap = laplacian_filter * ft_image
    # Shift back the zero-frequency component
    Lap = np.fft.ifftshift(Lap)
    # Perform inverse Fourier Transform and take the real part
    return np.real(np.fft.ifft2(Lap))


def scale_to_range(image, new_min=-1, new_max=1):
    """
    Scale the image values to a new range.

    Args:
    image (numpy.ndarray): Input image array.
    new_min (float): Minimum value of the new range.
    new_max (float): Maximum value of the new range.

    Returns:
    numpy.ndarray: Scaled image array.
    """
    old_min, old_max = np.min(image), np.max(image)
    old_range = old_max - old_min
    new_range = new_max - new_min
    return (((image - old_min) * new_range) / old_range) + new_min


def enhance_image(original, laplacian, c=-1):
    """
    Enhance the original image using the Laplacian-filtered image.

    Args:
    original (numpy.ndarray): Original image array.
    laplacian (numpy.ndarray): Laplacian-filtered image array.
    c (float): Enhancement factor.

    Returns:
    numpy.ndarray: Enhanced image array.
    """
    # Combine the original image with the Laplacian-filtered image
    enhanced = original + c * laplacian
    # Clip values to ensure they remain in the range [0, 1]
    return np.clip(enhanced, 0, 1)


def laplacian_filter_workflow(image_path, save_to_dir=False):
    """
    Main workflow for applying the Laplacian filter and enhancing the image.

    Args:
    image_path (str): Path to the input image file.
    save_to_dir (bool): Whether to save output images to directory.
    """
    # Load and normalize the image
    f = load_and_normalize_image(image_path)
    display_image(f, "Original_Image", image_path, save_to_dir)

    # Perform Fourier Transform
    F = fourier_transform(f)
    ft_magnitude = np.log1p(np.abs(F))
    display_image(ft_magnitude, "Fourier_Transform_Magnitude", image_path, save_to_dir)

    # Create and apply Laplacian filter
    H = create_laplacian_filter(F.shape)
    display_image(H, "Laplacian_Filter", image_path, save_to_dir)

    Lap = apply_laplacian_filter(F, H)
    LapScaled = scale_to_range(Lap)
    display_image(LapScaled, "Laplacian_Filtered_Image", image_path, save_to_dir)

    # Enhance the image
    g = enhance_image(f, LapScaled)
    display_image(g, "Enhanced_Image", image_path, save_to_dir)


