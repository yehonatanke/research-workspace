from typing import List
import numpy as np
import matplotlib.pyplot as plt


def fill_color_channel(n: int, color: str = 'red', save_to_dir: bool = False, color_number: int = 1) -> np.ndarray:
    """
    Create and display a series of matrices representing incrementally filled color channels.

    Parameters:
    n (int): Number of steps in the color fill process
    color (str): Color(s) to fill. Can be 'red', 'green', 'blue', or any combination
    save_to_dir (bool): Indicator for image-saving
    color_number (int): Unique number for this color combination

    Returns:
    np.ndarray: The final image array
    """
    I: np.ndarray = np.zeros((64, 64, 3))

    channels: list[int] = []
    if 'red' in color.lower():
        channels.append(0)
    if 'green' in color.lower():
        channels.append(1)
    if 'blue' in color.lower():
        channels.append(2)

    if not channels:
        raise ValueError("Please choose at least one color: 'red', 'green', or 'blue'")

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for i in range(1, n + 1):
        for channel in channels:
            I[:, :, channel] = i / n

        # Display the matrix instead of the image
        axes[i - 1].imshow(I[:5, :5], cmap='RdYlBu', vmin=0, vmax=1)
        axes[i - 1].set_xticks(np.arange(5))
        axes[i - 1].set_yticks(np.arange(5))
        axes[i - 1].set_xticklabels([])
        axes[i - 1].set_yticklabels([])

        # Add text annotations to show matrix values
        for ii in range(5):
            for jj in range(5):
                # Calculate the brightness of the cell
                brightness = np.mean(I[ii, jj])
                # Choose white text for dark backgrounds, black for light
                text_color = "white" if brightness < 0.5 else "black"

                text = axes[i - 1].text(jj, ii, f'[{I[ii, jj, 0]:.1f}\n{I[ii, jj, 1]:.1f}\n{I[ii, jj, 2]:.1f}]',
                                        ha="center", va="center", color=text_color, fontsize=6)

        axes[i - 1].set_title(f"Step {i}/{n}")

    plt.tight_layout()
    if save_to_dir:
        plt.savefig(f'output/matrix_output_{color_number:02d}.png', bbox_inches='tight')
    plt.show()

    return I


def fill_color_channel_v1(n: int, color: str = 'red', save_to_dir: bool = False, color_number: int = 1) -> np.ndarray:
    """
    Create and display a series of images with incrementally filled color channels.
    
    Parameters:
    n (int): Number of steps in the color fill process
    color (str): Color(s) to fill. Can be 'red', 'green', 'blue', or any combination
    save_to_dir (bool): Indicator for image-saving
    color_number (int): Unique number for this color combination

    Returns:
    np.ndarray: The final image array
    """
    # Create a 3D array of zeros with shape (64, 64, 3)
    # This represents a 64x64 pixel image with 3 color channels (RGB)
    I: np.ndarray = np.zeros((64, 64, 3))

    # Determine which color channels to fill based on the 'color' parameter
    channels: List[int] = []
    if 'red' in color.lower():
        channels.append(0)  # Red is the first channel (index 0)
    if 'green' in color.lower():
        channels.append(1)  # Green is the second channel (index 1)
    if 'blue' in color.lower():
        channels.append(2)  # Blue is the third channel (index 2)

    if not channels:
        raise ValueError("Please choose at least one color: 'red', 'green', or 'blue'")

    # Create a figure with n subplots, each representing a step in the process
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]  # Convert axes to a list if there's only one subplot

    # Fill the chosen color channels and display the image at each step
    for i in range(1, n + 1):
        for channel in channels:
            # Fill the channel with a value between 0 and 1
            # i/n ensures we get n even steps from 0 to 1
            I[:, :, channel] = i / n

        # Display the current image
        axes[i - 1].imshow(I)
        axes[i - 1].axis('off')  # Remove axis ticks and labels
        axes[i - 1].set_title(f"Step {i}/{n}")

    plt.tight_layout()
    if save_to_dir:
        plt.savefig(f'output/output_{color_number:02d}.png', bbox_inches='tight')
    plt.show()

    return I


# Usage examples
n: int = 14

# Red
fill_color_channel(n, 'red', color_number=1)  # Red
fill_color_channel(n, 'green', color_number=2)  # Green
fill_color_channel(n, 'blue', color_number=3)  # Blue
fill_color_channel(n, 'red green', color_number=4)  # Yellow
fill_color_channel(n, 'red blue', color_number=5)  # Purple
fill_color_channel(n, 'green blue', color_number=6)  # Cyan
fill_color_channel(n, 'red green blue', color_number=7)  # White

"""
Explanation of the image array and axes:

1. Image Array (I):
   - Shape: (64, 64, 3)
   - First axis (64): Represents the height of the image (64 pixels)
   - Second axis (64): Represents the width of the image (64 pixels)
   - Third axis (3): Represents the color channels (Red, Green, Blue)

2. Color Channels:
   - I[:,:,0]: Red channel
   - I[:,:,1]: Green channel
   - I[:,:,2]: Blue channel

3. Color Values:
   - Range from 0 (no color) to 1 (full color intensity)
   - In each step, we fill the chosen channels with i/n, where i is the current step
     and n is the total number of steps. This creates a gradual fill effect.

4. Displayed Images:
   - We create n subplots, each showing the image at a different step of the process
   - The x-axis of each subplot represents the width of the image
   - The y-axis of each subplot represents the height of the image
   - Color intensity is represented by the values in the color channels

5. Color Mixing:
   - When multiple channels are chosen, they mix to create new colors:
     * Red + Green = Yellow
     * Red + Blue = Purple
     * Green + Blue = Cyan
     * Red + Green + Blue = White (or shades of gray)

This visualization helps to understand how digital images are composed of 
separate color channels, and how these channels combine to create different colors.
"""
