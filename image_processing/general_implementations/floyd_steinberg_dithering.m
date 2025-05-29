function floyd_steinberg_dithering(input_image_path, output_image_path)
    % Applies Floyd-Steinberg dithering to a grayscale image.
    %
    % Args:
    % input_image_path: Path to the grayscale input image.
    % output_image_path: Path to save the dithered image.
    
    % Read the image and convert it to a grayscale double matrix
    img = imread(input_image_path);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = double(img) / 255;

    [height, width] = size(img);
    
    % Floyd-Steinberg dithering algorithm
    for y = 1:height
        for x = 1:width
            old_pixel = img(y, x);
            new_pixel = round(old_pixel);
            img(y, x) = new_pixel;
            error = old_pixel - new_pixel;
            
            % Distribute the error to neighboring pixels
            if x + 1 <= width
                img(y, x + 1) = img(y, x + 1) + error * (7 / 16);
            end
            if x - 1 >= 1 && y + 1 <= height
                img(y + 1, x - 1) = img(y + 1, x - 1) + error * (3 / 16);
            end
            if y + 1 <= height
                img(y + 1, x) = img(y + 1, x) + error * (5 / 16);
            end
            if x + 1 <= width && y + 1 <= height
                img(y + 1, x + 1) = img(y + 1, x + 1) + error * (1 / 16);
            end
        end
    end

    % Convert back to uint8 and save the dithered image
    imwrite(uint8(img * 255), output_image_path);
end

input_image_path = 'input_image.png';  % Input image path
output_image_path = 'dithered_output.png';  % Output image path

floyd_steinberg_dithering(input_image_path, output_image_path);
