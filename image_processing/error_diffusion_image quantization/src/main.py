from error_diffusion import process_image


def main():
    image_path = 'input/tungsten_original.JPG'

    for m in range(2, 6):
        process_image(image_path, m)


if __name__ == "__main__":
    main()
