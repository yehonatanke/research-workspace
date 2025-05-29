% Read an image
img = imread('input/im1.png');  % Replace with your image file

% Display the original image
figure;
imshow(img);
title('Original Image');

% Convert the image to grayscale
gray_img = rgb2gray(img);

% Display the grayscale image
figure;
imshow(gray_img);
title('Grayscale Image');

edges = edge(gray_img, 'Canny');
figure;
imshow(edges);
title('Edge Detected Image');

% Histogram 
figure;
imhist(gray_img);
title('Histogram of Grayscale Image');

imwrite(edges, 'edges_output.jpg');  % Save the edge-detected image
