% Edge Detection using Sobel and Canny filters 

% Read the image and convert it to grayscale
img = imread('input_image.jpg');
gray_img = rgb2gray(img);

% Display original image
figure, imshow(gray_img);
title('Original Grayscale Image');

%% Sobel Edge Detection
sobel_edges = edge(gray_img, 'Sobel');

% Display Sobel edge-detected image
figure, imshow(sobel_edges);
title('Sobel Edge Detection');

%% Canny Edge Detection
canny_edges = edge(gray_img, 'Canny');

% Display Canny edge-detected image
figure, imshow(canny_edges);
title('Canny Edge Detection');

% Save 
imwrite(sobel_edges, 'sobel_edges.jpg');
imwrite(canny_edges, 'canny_edges.jpg');
