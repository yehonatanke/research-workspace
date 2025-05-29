import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_3d(image, ax):
    """Plot a 3D surface of the image intensity."""
    x = np.arange(0, image.shape[0])
    y = np.arange(0, image.shape[1])
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, image, cmap='viridis')
    ax.set_zlim(0, 300)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')


def display_results(images, titles):
    """
    Display multiple images with their 3D plots.

    Args:
    images (list): List of image arrays.
    titles (list): List of titles for each image.
    """
    n = len(images)
    fig = plt.figure(figsize=(20, 10 * n))

    for i, (image, title) in enumerate(zip(images, titles)):
        # 2D image
        ax1 = fig.add_subplot(n, 2, 2 * i + 1)
        ax1.imshow(image, cmap='gray')
        ax1.set_title(f'{title} - 2D')
        ax1.axis('off')

        # 3D plot
        ax2 = fig.add_subplot(n, 2, 2 * i + 2, projection='3d')
        plot_3d(image, ax2)
        ax2.set_title(f'{title} - 3D')

    plt.tight_layout()
    plt.show()
