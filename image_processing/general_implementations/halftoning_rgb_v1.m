function halftoning_rgb_example(image_path)
    % Read an image
    img = imread('input/im1');
    
    % Apply Floyd-Steinberg error diffusion
    img_error_diffusion = floyd_steinberg_error_diffusion(img);
    figure, imshow(img_error_diffusion), title('Error Diffusion (Floyd-Steinberg)');
    
    % Apply Ordered Dithering
    img_ordered_dithering = ordered_dithering(img);
    figure, imshow(img_ordered_dithering), title('Ordered Dithering (Bayer Matrix)');
end

%% Floyd-Steinberg Error Diffusion Function
function output = floyd_steinberg_error_diffusion(img)
    % Convert image to double for calculation
    img = double(img);
    [rows, cols, channels] = size(img);
    
    % Floyd-Steinberg diffusion matrix coefficients
    diffusion_matrix = [0 0 7; 3 5 1] / 16;
    
    % Loop over each pixel in the image
    for y = 1:rows
        for x = 1:cols
            for c = 1:channels
                % Get original pixel value
                old_pixel = img(y, x, c);
                
                % Threshold to either 0 or 255
                new_pixel = round(old_pixel / 255) * 255;
                img(y, x, c) = new_pixel;
                
                % Compute error
                error = old_pixel - new_pixel;
                
                % Diffuse error to neighboring pixels
                for dy = 0:1
                    for dx = -1:1
                        if (x + dx > 0) && (x + dx <= cols) && (y + dy <= rows)
                            img(y + dy, x + dx, c) = img(y + dy, x + dx, c) + error * diffusion_matrix(dy + 1, dx + 2);
                        end
                    end
                end
            end
        end
    end
    
    % Convert the image back to uint8 and return
    output = uint8(img);
end

%% Ordered Dithering Function
function output = ordered_dithering(img)
    % Convert image to double for calculation
    img = double(img);
    
    % Define a 4x4 Bayer matrix
    bayer_matrix = [0 8 2 10; 12 4 14 6; 3 11 1 9; 15 7 13 5] * (255 / 16);
    [rows, cols, channels] = size(img);
    [bm_rows, bm_cols] = size(bayer_matrix);
    
    % Loop over each pixel and apply the Bayer matrix
    for y = 1:rows
        for x = 1:cols
            for c = 1:channels
                % Get the threshold from the Bayer matrix
                threshold = bayer_matrix(mod(y-1, bm_rows) + 1, mod(x-1, bm_cols) + 1);
                
                % Apply thresholding
                if img(y, x, c) > threshold
                    img(y, x, c) = 255;
                else
                    img(y, x, c) = 0;
                end
            end
        end
    end
    
    % Convert the image back to uint8 and return
    output = uint8(img);
end
