% Read the image and convert it to grayscale
img = imread('input_image.jpg');
gray_img = rgb2gray(img);

% Display original image
figure, imshow(gray_img);
title('Original Grayscale Image');

% Perform 2D Fourier Transform
F = fft2(double(gray_img));

% Shift the zero-frequency component to the center
F_shifted = fftshift(F);

% Display magnitude spectrum
magnitude_spectrum = log(abs(F_shifted) + 1);
figure, imshow(magnitude_spectrum, []);
title('Magnitude Spectrum of Fourier Transform');

% Create a low-pass filter
[rows, cols] = size(gray_img);
radius = 50; % Adjust radius for filtering effect
[X, Y] = meshgrid(1:cols, 1:rows);
center_x = cols / 2;
center_y = rows / 2;
low_pass_filter = sqrt((X - center_x).^2 + (Y - center_y).^2) <= radius;

% Apply the low-pass filter to the shifted Fourier transform
F_filtered_shifted = F_shifted .* low_pass_filter;

% Shift back and perform the inverse Fourier transform
F_filtered = ifftshift(F_filtered_shifted);
filtered_img = ifft2(F_filtered);

% Display filtered image (real part only)
filtered_img = real(filtered_img);
figure, imshow(filtered_img, []);
title('Low-Pass Filtered Image');

% Save filtered image
imwrite(uint8(filtered_img), 'filtered_image.jpg');
