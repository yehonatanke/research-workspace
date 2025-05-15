function I = load_image(file_path)
I = imread(file_path);
if size(I,3) == 3
    I = rgb2gray(I);
end
I = im2double(I);
end

function I_filt = apply_gaussian_filter(I, sigma)
h = fspecial('gaussian', [5 5], sigma);
I_filt = imfilter(I, h, 'replicate');
end

function I_edge = detect_edges(I, method, thresh)
switch method
    case 'sobel'
        I_edge = edge(I, 'sobel', thresh);
    case 'canny'
        I_edge = edge(I, 'canny', thresh);
    case 'prewitt'
        I_edge = edge(I, 'prewitt', thresh);
    otherwise
        I_edge = edge(I, 'canny', thresh);
end
end

function [I_bin, thresh] = binarize_image(I, method)
switch method
    case 'otsu'
        thresh = graythresh(I);
        I_bin = imbinarize(I, thresh);
    case 'adaptive'
        I_bin = imbinarize(I, 'adaptive');
    otherwise
        thresh = graythresh(I);
        I_bin = imbinarize(I, thresh);
end
end

function I_enh = enhance_contrast(I, method)
switch method
    case 'histeq'
        I_enh = histeq(I);
    case 'adapthisteq'
        I_enh = adapthisteq(I);
    case 'imadjust'
        I_enh = imadjust(I);
    otherwise
        I_enh = histeq(I);
end
end

function features = extract_features(I)
stats = regionprops('table', I, 'Area', 'Perimeter', 'MeanIntensity', 'MajorAxisLength', 'MinorAxisLength');
features = table2array(stats);
end

function I_seg = segment_image(I, method, k)
switch method
    case 'kmeans'
        [idx, ~] = kmeans(I(:), k);
        I_seg = reshape(idx, size(I));
    case 'watershed'
        I_seg = watershed(I);
    otherwise
        [idx, ~] = kmeans(I(:), k);
        I_seg = reshape(idx, size(I));
end
end

function [freq, power] = frequency_analysis(I)
F = fft2(I);
F_shift = fftshift(F);
power = abs(F_shift).^2;
freq = linspace(-0.5, 0.5, size(I,1));
end

function I_denoised = denoise_image(I, method)
switch method
    case 'median'
        I_denoised = medfilt2(I, [3 3]);
    case 'wiener'
        I_denoised = wiener2(I, [5 5]);
    case 'wavelet'
        I_denoised = wdenoise2(I);
    otherwise
        I_denoised = medfilt2(I, [3 3]);
end
end

function metrics = evaluate_segmentation(I_seg, I_gt)
I_seg = logical(I_seg);
I_gt = logical(I_gt);
TP = sum(I_seg(:) & I_gt(:));
FP = sum(I_seg(:) & ~I_gt(:));
FN = sum(~I_seg(:) & I_gt(:));
TN = sum(~I_seg(:) & ~I_gt(:));
metrics.accuracy = (TP + TN) / (TP + TN + FP + FN);
metrics.precision = TP / (TP + FP);
metrics.recall = TP / (TP + FN);
metrics.f1 = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall);
end
