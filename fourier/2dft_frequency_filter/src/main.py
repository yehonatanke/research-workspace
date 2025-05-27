import os
import argparse
from image_processor import process_image


def main_v2():
    parser = argparse.ArgumentParser(description="Image processing with Fourier transforms")
    parser.add_argument("image_path", help="Path to the input image file")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to output folder")
    parser.add_argument("--output_folder", default="output", help="Folder to save plots (default: output)")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return

    try:
        process_image(args.image_path, save_plots=args.save_plots, output_folder=args.output_folder)
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def main_v1():
    image_path = "input/frogs.png"
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return

    try:
        process_image(image_path)
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def main():
    image_path = "/..."
    save_plots = False
    output_folder = "output"

    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return

    try:
        process_image(image_path, save_plots=save_plots, output_folder=output_folder)
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
