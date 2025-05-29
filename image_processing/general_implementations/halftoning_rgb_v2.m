function halftoning_techniques()
    input_image = imread('input/im1.png');

    % Apply error diffusion
    error_diffusion_result = error_diffusion(input_image);
    imwrite(error_diffusion_result, 'error_diffusion_result.png');

    % Apply dithering
    dithering_result = ordered_dithering(input_image);
    imwrite(dithering_result, 'dithering_result.png');
end

function output = error_diffusion(image)
    % Apply error diffusion halftoning to an RGB image
    %
    % Args:
    % image: Input RGB image (height x width x 3 uint8 array)
    %
    % Returns:
    % output: Halftoned image (height x width x 3 uint8 array)

    [height, width, ~] = size(image);
    output = zeros(size(image), 'uint8');
    error = zeros(size(image), 'double');
    threshold = 128;

    for y = 1:height
        for x = 1:width
            old_pixel = double(image(y, x, :)) + error(y, x, :);
            new_pixel = uint8(old_pixel > threshold) * 255;
            output(y, x, :) = new_pixel;

            quant_error = old_pixel - double(new_pixel);

            if x < width
                error(y, x+1, :) = error(y, x+1, :) + quant_error * 7/16;
            end
            if x > 1 && y < height
                error(y+1, x-1, :) = error(y+1, x-1, :) + quant_error * 3/16;
            end
            if y < height
                error(y+1, x, :) = error(y+1, x, :) + quant_error * 5/16;
            end
            if x < width && y < height
                error(y+1, x+1, :) = error(y+1, x+1, :) + quant_error * 1/16;
            end
        end
    end
end

function output = ordered_dithering(image)
    % Apply ordered dithering halftoning to an RGB image
    %
    % Args:
    % image: Input RGB image (height x width x 3 uint8 array)
    %
    % Returns:
    % output: Halftoned image (height x width x 3 uint8 array)

    [height, width, ~] = size(image);
    output = zeros(size(image), 'uint8');

    % Bayer matrix 4x4
    bayer_matrix = [
        0,  8,  2, 10;
        12, 4, 14,  6;
        3, 11,  1,  9;
        15, 7, 13,  5
    ] / 16;

    for y = 1:height
        for x = 1:width
            threshold = bayer_matrix(mod(y-1, 4) + 1, mod(x-1, 4) + 1) * 255;
            output(y, x, :) = uint8(image(y, x, :) > threshold) * 255;
        end
    end
end
