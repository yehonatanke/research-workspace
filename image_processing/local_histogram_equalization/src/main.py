from image_processing import load_image, local_histogram_equalization
from visualization import plot_images_and_histograms


def main():
    # Load the embedded_squares image
    image_path = 'input/embedded_squares.JPG'
    try:
        image = load_image(image_path)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Apply local histogram equalization
    enhanced_image = local_histogram_equalization(image, kernel_size=15)

    # Display original and enhanced images
    plot_images_and_histograms(image, enhanced_image)


if __name__ == "__main__":
    main()
