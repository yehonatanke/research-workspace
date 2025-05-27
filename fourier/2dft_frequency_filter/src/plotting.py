import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fftshift, ifft2, ifftshift


def plot_results(x_filt, x_blue_filt, x_blue, filt, scene, save_plots=False, output_folder='output'):
    """
    Plot the results of the image processing.

    Args:
    x_filt (np.array): Filtered image.
    x_blue_filt (np.array): Filtered blue channel.
    x_blue (np.array): Original blue channel.
    filt (np.array): Applied filter.
    scene (int): Current scene number.
    save_plots (bool): Whether to save the plots to files.
    output_folder (str): Folder to save the plots in.
    """
    if save_plots and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plt.figure(1)
    plt.imshow(x_filt.astype(np.uint8))
    plt.title('Filtered Image', fontsize=20)
    if save_plots:
        plt.savefig(os.path.join(output_folder, f'scene_{scene}_filtered_image.png'))
    plt.close()

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(0, x_blue_filt.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = np.abs(fftshift(x_blue_filt))
    ax.plot_surface(X, Y, Z)
    ax.set_zlim(0, 10 ** 6)
    plt.title('abs(Filtered FT of Image)', fontsize=20)
    if save_plots:
        plt.savefig(os.path.join(output_folder, f'scene_{scene}_filtered_ft.png'))
    plt.close()

    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    Z = np.abs(fftshift(x_blue - x_blue_filt))
    ax.plot_surface(X, Y, Z)
    plt.title('abs(Values Removed from FT of Image)', fontsize=20)
    if save_plots:
        plt.savefig(os.path.join(output_folder, f'scene_{scene}_removed_values.png'))
    plt.close()

    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')
    filt_image = np.real(ifftshift(ifft2(filt)))
    x = y = np.arange(0, filt_image.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = filt_image
    ax.plot_surface(X, Y, Z)
    plt.title('abs(The filter in the Image domain)', fontsize=20)
    if save_plots:
        plt.savefig(os.path.join(output_folder, f'scene_{scene}_filter_image_domain.png'))
    plt.close()

    fig = plt.figure(5)
    ax = fig.add_subplot(111, projection='3d')
    Z = fftshift(np.real(filt))
    ax.plot_surface(X, Y, Z)
    plt.title('abs(The filter in the FT domain)', fontsize=20)
    if save_plots:
        plt.savefig(os.path.join(output_folder, f'scene_{scene}_filter_ft_domain.png'))
    plt.close()

    if scene == 10:
        fig = plt.figure(6)
        ax = fig.add_subplot(111, projection='3d')
        Z = np.abs(ifft2(x_blue_filt)).T
        ax.plot_surface(X, Y, Z)
        ax.view_init(90, 90)
        plt.title('Special view for Scene 10', fontsize=20)
        if save_plots:
            plt.savefig(os.path.join(output_folder, 'scene_10_special_view.png'))
        plt.close()

    plt.show()
