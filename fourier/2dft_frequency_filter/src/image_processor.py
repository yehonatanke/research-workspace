import numpy as np
from PIL import Image
from fourier_utils import apply_fourier_transform, create_filter, apply_filter
from image_utils import create_filtered_image
from plotting import plot_results


def process_image(image_path, im_size=600, im_start=23, save_plots=False, output_folder='output'):
    """
    Process the image using Fourier transforms.

    Args:
    image_path (str): Path to the input image file.
    im_size (int): Size of the image to process (default: 600).
    im_start (int): Starting point for image cropping (default: 23).
    save_plots (bool): Whether to save the plots to files.
    output_folder (str): Folder to save the plots in.
    """
    if im_size <= 0 or im_start < 0:
        raise ValueError("im_size must be positive and im_start must be non-negative")

    try:
        with Image.open(image_path) as img:
            x = np.array(img)
    except Exception as e:
        raise IOError(f"Error loading image: {str(e)}")

    # Ensure the image is large enough
    if x.shape[0] < im_size + im_start or x.shape[1] < im_size:
        raise ValueError("Image is too small for the specified im_size and im_start")

    # Crop and pad the image to ensure it's square
    x_cropped = x[im_start:im_start + im_size, :im_size]
    if x_cropped.shape[1] < im_size:
        padding = [(0, 0), (0, im_size - x_cropped.shape[1]), (0, 0)]
        x_cropped = np.pad(x_cropped, padding, mode='constant')

    x_red = x_cropped[:, :, 0]
    x_green = x_cropped[:, :, 1]
    x_blue = x_cropped[:, :, 2]

    for scene in range(1, 11):
        print(f"Processing scene {scene}")
        filt_l, filt_domain, img_ft = set_scene_parameters(scene)

        x_red_ft, x_green_ft, x_blue_ft = apply_fourier_transform(x_red, x_green, x_blue, img_ft)

        filt = create_filter(x_red_ft.shape, filt_l, filt_domain, x_blue_ft)
        x_red_filt, x_green_filt, x_blue_filt = apply_filter(x_red_ft, x_green_ft, x_blue_ft, filt)

        x_filt = create_filtered_image(x_red_filt, x_green_filt, x_blue_filt, x_cropped, im_size)

        plot_results(x_filt, x_blue_filt, x_blue_ft, filt, scene, save_plots, output_folder)


def set_scene_parameters(scene):
    """
    Set filter parameters based on the current scene.

    Args:
    scene (int): Current scene number (1-10).

    Returns:
    tuple: (filt_l, filt_domain, img_ft)
    """
    filt_l, filt_domain, img_ft = 298, 2, 1
    if scene == 2:
        filt_l = 100
    elif scene == 3:
        filt_l = 50
    elif scene == 4:
        filt_l = 10
    elif scene == 5:
        filt_l, filt_domain = 50, 3
    elif scene == 6:
        filt_l, filt_domain = 100, 6
    elif scene == 7:
        filt_l = 500
    elif scene == 8:
        filt_l = 1000
    elif scene == 9:
        filt_l, img_ft = 0.001, 2
    elif scene == 10:
        img_ft = 3
    return filt_l, filt_domain, img_ft
