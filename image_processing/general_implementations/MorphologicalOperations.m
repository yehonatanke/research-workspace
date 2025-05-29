% Morphological Operations
% - Dilation: Expands the white regions in the image.
% - Erosion: Shrinks the white regions.
% - Opening: Removes small objects from the foreground (smooths boundaries).
% - Closing: Fills small holes in the foreground.
% - Morphological Gradient: Highlights the edges.
% - Top-Hat and Bottom-Hat transformations.

% Read the image (grayscale or binary for morphology)
inputImage = imread('your_image.png'); % Load the image
if size(inputImage, 3) == 3
    % Convert RGB to grayscale if the input is a colored image
    inputImage = rgb2gray(inputImage);
end

% Display the original image
figure, imshow(inputImage);
title('Original Image');

% Define the structuring element (strel) - a key element for morphological operations
se = strel('disk', 5); % Disk-shaped structuring element with radius 5

%% Dilation
dilatedImage = imdilate(inputImage, se);
figure, imshow(dilatedImage);
title('Dilated Image');

%% Erosion
erodedImage = imerode(inputImage, se);
figure, imshow(erodedImage);
title('Eroded Image');

%% Opening (Erosion followed by Dilation)
openedImage = imopen(inputImage, se);
figure, imshow(openedImage);
title('Opened Image');

%% Closing (Dilation followed by Erosion)
closedImage = imclose(inputImage, se);
figure, imshow(closedImage);
title('Closed Image');

%% Morphological Gradient (Difference between Dilation and Erosion)
gradientImage = imsubtract(dilatedImage, erodedImage);
figure, imshow(gradientImage);
title('Morphological Gradient');

%% Top-Hat Transformation (Original minus Opening)
topHatImage = imtophat(inputImage, se);
figure, imshow(topHatImage);
title('Top-Hat Transformation');

%% Bottom-Hat Transformation (Closing minus Original)
bottomHatImage = imbothat(inputImage, se);
figure, imshow(bottomHatImage);
title('Bottom-Hat Transformation');

% Save results
imwrite(dilatedImage, 'dilatedImage.png');
imwrite(erodedImage, 'erodedImage.png');
imwrite(openedImage, 'openedImage.png');
imwrite(closedImage, 'closedImage.png');
imwrite(gradientImage, 'gradientImage.png');
imwrite(topHatImage, 'topHatImage.png');
imwrite(bottomHatImage, 'bottomHatImage.png');

