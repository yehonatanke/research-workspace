function restored_image = restore_image(input_image, psf, noise_power)
% RESTORE_IMAGE Perform image restoration using Wiener filtering
%
% Inputs:
%   input_image - The degraded input image (grayscale)
%   psf - Point Spread Function (PSF) of the blur
%   noise_power - Estimate of the noise power
%
% Output:
%   restored_image - The restored image

% Check input arguments
narginchk(3, 3);
validateattributes(input_image, {'numeric'}, {'2d', 'nonempty', 'real', 'nonsparse'});
validateattributes(psf, {'numeric'}, {'2d', 'nonempty', 'real', 'nonsparse'});
validateattributes(noise_power, {'numeric'}, {'scalar', 'real', 'nonnegative'});

% Ensure the input image is in double precision
input_image = im2double(input_image);

% Compute the Fourier transform of the input image and PSF
F = fft2(input_image);
H = psf2otf(psf, size(input_image));

% Compute the power spectrum of the true image (approximation)
power_spectrum = abs(F).^2;

% Wiener filter in the frequency domain
G = conj(H) ./ (abs(H).^2 + noise_power ./ power_spectrum);

% Apply the Wiener filter
restored_F = G .* F;

% Inverse Fourier transform to get the restored image
restored_image = real(ifft2(restored_F));

% Clip values to [0, 1] range
restored_image = max(0, min(1, restored_image));

end

% usage
original_image = imread('input/cameraman.tif');

% Create a motion blur PSF
len = 21;
theta = 11;
PSF = fspecial('motion', len, theta);

% Add some noise to the blurred image
blurred = imfilter(original_image, PSF, 'conv', 'circular');
noise_var = 0.0001;
noisy_blurred = imnoise(blurred, 'gaussian', 0, noise_var);

% Estimate the noise power
noise_power = noise_var * prod(size(original_image));

% Perform image restoration
restored = restore_image(noisy_blurred, PSF, noise_power);

% Display results
figure;
subplot(2,2,1), imshow(original_image), title('Original Image');
subplot(2,2,2), imshow(noisy_blurred), title('Degraded Image');
subplot(2,2,3), imshow(restored), title('Restored Image');
subplot(2,2,4), imshow(abs(original_image - restored), []), title('Restoration Error');
