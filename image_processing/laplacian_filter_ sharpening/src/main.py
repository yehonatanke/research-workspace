from pathlib import Path
from laplacian_frequency_filter import laplacian_filter_workflow


def process_images(image_directory, save_output=True):
    """
    Process all images in the specified directory using the Laplacian filter workflow.

    Args:
    image_directory (str or Path): Path to the directory containing input images.
    save_output (bool): Whether to save the output images or not.
    """
    # Convert image_directory to a Path object
    image_dir = Path(image_directory)

    # Ensure the output directory exists
    if save_output:
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)

    # Get all image files in the directory
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]

    # Process each image
    for image_file in image_files:
        print(f"Processing {image_file.name}...")
        laplacian_filter_workflow(str(image_file), save_to_dir=save_output)
        print(f"Finished processing {image_file.name}")


if __name__ == "__main__":
    input_directory = Path("/input")

    # Run the workflow on all images
    process_images(input_directory, save_output=True)
