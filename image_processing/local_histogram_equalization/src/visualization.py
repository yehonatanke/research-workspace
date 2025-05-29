import matplotlib.pyplot as plt


def plot_images_and_histograms(original_image, enhanced_image):
    plt.figure(figsize=(12, 6))

    plt.subplot(221)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(222)
    plt.hist(original_image.flatten(), 256, [0, 256])
    plt.title('Original Histogram')

    plt.subplot(223)
    plt.imshow(enhanced_image, cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.subplot(224)
    plt.hist(enhanced_image.flatten(), 256, [0, 256])
    plt.title('Enhanced Histogram')

    plt.tight_layout()
    plt.show()
