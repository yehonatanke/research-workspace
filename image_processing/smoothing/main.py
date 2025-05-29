from image_processing import create_circle_image, create_square_image, create_triangle_image, smooth_image
from visualization import display_results


def process_and_display(create_func, shape_name, **kwargs):
    """Process a shape and display results."""
    original = create_func(**kwargs)
    smoothed = smooth_image(original)
    display_results([original, smoothed], [f'Original {shape_name}', f'Smoothed {shape_name}'])


def main():
    """Main function to run the image processing pipeline."""
    # Process circle
    process_and_display(create_circle_image, 'Circle', size=100, radius=30)

    # Process square
    process_and_display(create_square_image, 'Square', size=100, side=60)

    # Process triangle
    process_and_display(create_triangle_image, 'Triangle', size=100, side=70)

    # Compare all shapes
    circle = create_circle_image(size=100, radius=30)
    square = create_square_image(size=100, side=60)
    triangle = create_triangle_image(size=100, side=70)

    all_shapes = [circle, square, triangle]
    all_smoothed = [smooth_image(shape) for shape in all_shapes]

    display_results(all_shapes + all_smoothed,
                    ['Original Circle', 'Original Square', 'Original Triangle',
                     'Smoothed Circle', 'Smoothed Square', 'Smoothed Triangle'])


if __name__ == "__main__":
    main()
