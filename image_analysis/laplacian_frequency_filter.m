function filtered_image = laplacian_frequency_filter(input_image)
    if ischar(input_image)
        img = imread(input_image);
    else
        img = input_image;
    end
    
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    img = im2double(img);
    
    [M, N] = size(img);
    
    [X, Y] = meshgrid(1:N, 1:M);
    
    X = X - N/2;
    Y = Y - M/2;
    
    D = X.^2 + Y.^2;
    H = -4 * pi^2 * D;
    
    F = fftshift(fft2(img));
    
    filtered_F = F .* H;
    
    filtered_image = real(ifft2(ifftshift(filtered_F)));
    
    filtered_image = (filtered_image - min(filtered_image(:))) / (max(filtered_image(:)) - min(filtered_image(:)));
end
