function [segmented_image, cluster_centers] = kmeansImageSegmentation(image, k, max_iterations)
% KMEANSIMAGESEGMENTATION Segments an image using K-means clustering
%   [segmented_image, cluster_centers] = kmeansImageSegmentation(image, k, max_iterations)
%
%   Inputs:
%       image - RGB image (3D matrix)
%       k - Number of clusters (segments)
%       max_iterations - Maximum number of iterations for K-means
%
%   Outputs:
%       segmented_image - Segmented image (each pixel replaced with its cluster center)
%       cluster_centers - Final cluster centers

% Ensure the input image is in double format
image = im2double(image);

% Reshape the image to a 2D array of pixels
[height, width, channels] = size(image);
pixels = reshape(image, height * width, channels);

% Initialize cluster centers randomly
cluster_centers = rand(k, channels);

% Main K-means loop
for iter = 1:max_iterations
    % Compute distances between pixels and cluster centers
    distances = pdist2(pixels, cluster_centers);
    
    % Assign each pixel to the nearest cluster
    [~, pixel_labels] = min(distances, [], 2);
    
    % Store old cluster centers for convergence check
    old_centers = cluster_centers;
    
    % Update cluster centers
    for i = 1:k
        cluster_pixels = pixels(pixel_labels == i, :);
        if ~isempty(cluster_pixels)
            cluster_centers(i, :) = mean(cluster_pixels);
        end
    end
    
    % Check for convergence
    if sum(abs(old_centers(:) - cluster_centers(:))) < 1e-6
        break;
    end
end

% Create the segmented image
segmented_pixels = cluster_centers(pixel_labels, :);
segmented_image = reshape(segmented_pixels, height, width, channels);

% Visualize results
figure;

subplot(1,2,1);
imshow(image);
title('Original Image');

subplot(1,2,2);
imshow(segmented_image);
title(['Segmented Image (K = ', num2str(k), ')']);

% Display cluster centers as color swatches
figure;
for i = 1:k
    subplot(1, k, i);
    imshow(ones(50, 50, 3) .* reshape(cluster_centers(i, :), 1, 1, 3));
    title(['Cluster ', num2str(i)]);
end
sgtitle('Cluster Centers');

end

img = imread('output/im1.png');
k = 5; % Number of segments
max_iterations = 100;
[segmented_img, centers] = kmeansImageSegmentation(img, k, max_iterations);
