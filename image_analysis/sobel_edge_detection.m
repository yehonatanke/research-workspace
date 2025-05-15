function [edges, magnitude, direction] = sobelEdgeDetection(image)
% SOBELEDGEDETECTION Performs Sobel edge detection on the input image
%   [edges, magnitude, direction] = sobelEdgeDetection(image)
%
%   Input:
%       image - Grayscale image (2D matrix)
%
%   Outputs:
%       edges - Binary edge map
%       magnitude - Edge magnitude
%       direction - Edge direction (in radians)

% Ensure the input image is grayscale
if size(image, 3) > 1
    image = rgb2gray(image);
end

% Convert image to double for precise calculations
image = im2double(image);

% Define Sobel kernels
sobelX = [-1 0 1; -2 0 2; -1 0 1];
sobelY = [-1 -2 -1; 0 0 0; 1 2 1];

% Apply Sobel kernels
Gx = imfilter(image, sobelX, 'replicate');
Gy = imfilter(image, sobelY, 'replicate');

% Calculate gradient magnitude and direction
magnitude = sqrt(Gx.^2 + Gy.^2);
direction = atan2(Gy, Gx);

% Normalize magnitude to [0, 1] range
magnitude = magnitude / max(magnitude(:));

% Threshold the magnitude to get binary edge map
% You can adjust this threshold based on your needs
threshold = 0.1;
edges = magnitude > threshold;

% Visualize results
figure;

subplot(2,2,1);
imshow(image);
title('Original Image');

subplot(2,2,2);
imshow(edges);
title('Edge Detection');

subplot(2,2,3);
imshow(magnitude);
title('Edge Magnitude');

subplot(2,2,4);
imshow(direction, []);
title('Edge Direction');
colormap(gca, 'hsv');
colorbar;

end

img = imread('input/im1.png');
[edges, magnitude, direction] = sobelEdgeDetection(img);
