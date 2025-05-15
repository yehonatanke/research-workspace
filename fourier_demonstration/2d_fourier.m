function main()
    originalImage = imread('ImageSmall.jpg');
    processImage(originalImage);
end

function processImage(originalImage)
    % Process the image with various filters
    [redChannel, greenChannel, blueChannel] = extractColorChannels(originalImage);
    
    for sceneNumber = 1:10
        [filterLength, filterDomain, fourierTransformType] = getSceneParameters(sceneNumber);
        
        [redFT, greenFT, blueFT] = applyFourierTransform(redChannel, greenChannel, blueChannel, fourierTransformType);
        
        filterMask = createFilter(filterLength, filterDomain, size(redChannel));
        
        [filteredImage, filteredBlueFT] = applyFilter(redFT, greenFT, blueFT, filterMask);
        
        displayResults(filteredImage, filteredBlueFT, blueFT, filterMask);
        
        if sceneNumber == 10
            displayFinalResult(filteredImage);
        end
    end
end

function [redChannel, greenChannel, blueChannel] = extractColorChannels(image)
    % Extract color channels from the image
    imageSize = 600;
    startIndex = 23;
    redChannel = image(startIndex:imageSize+startIndex-1, :, 1);
    greenChannel = image(startIndex:imageSize+startIndex-1, :, 2);
    blueChannel = image(startIndex:imageSize+startIndex-1, :, 3);
end

function [filterLength, filterDomain, fourierTransformType] = getSceneParameters(sceneNumber)
    % Get parameters for each scene
    filterLength = 298;
    filterDomain = 2;
    fourierTransformType = 1;
    
    switch sceneNumber
        case 2
            filterLength = 100;
        case 3
            filterLength = 50;
        case 4
            filterLength = 10;
        case 5
            filterLength = 50;
            filterDomain = 3;
        case 6
            filterLength = 100;
            filterDomain = 6;
        case 7
            filterLength = 500;
        case 8
            filterLength = 1000;
        case 9
            filterLength = 0.001;
            fourierTransformType = 2;
        case 10
            fourierTransformType = 3;
    end
end

function [redFT, greenFT, blueFT] = applyFourierTransform(redChannel, greenChannel, blueChannel, fourierTransformType)
    % Apply Fourier Transform based on fourierTransformType parameter
    switch fourierTransformType
        case 1 % full image FT
            redFT = fft2(redChannel);
            greenFT = fft2(greenChannel);
            blueFT = fft2(blueChannel);
        case 2 % Filtering with only keeping the magnitude of the image FT
            redFT = abs(fft2(redChannel));
            greenFT = abs(fft2(greenChannel));
            blueFT = abs(fft2(blueChannel));
        case 3 % Filtering with only keeping the phase of the image FT
            redFT = exp(1i * angle(fft2(redChannel)));
            greenFT = exp(1i * angle(fft2(greenChannel)));
            blueFT = exp(1i * angle(fft2(blueChannel)));
    end
end

function filterMask = createFilter(filterLength, filterDomain, imageSize)
    % Create filter based on filterDomain parameter
    imageDimension = imageSize(1);
    
    switch filterDomain
        case 1 % LPF via a square function in the image domain
            filterMask = zeros(imageSize);
            filterMask(1:filterLength, 1:filterLength) = ones(filterLength, filterLength) / (filterLength^2);
            filterMask = fft2(filterMask);
        case 2 % Square LPF in the Freq domain
            filterMask = createSquareLPF(imageDimension, filterLength);
        case 3 % Square HPF in the Freq domain
            filterMask = createSquareHPF(imageDimension, filterLength);
        case 4 % Circular LPF in the Freq domain
            filterMask = createCircularLPF(imageDimension, filterLength);
        case 5 % Circular HPF in the Freq domain
            filterMask = createCircularHPF(imageDimension, filterLength);
        case 6 % Threshold
            filterMask = createThresholdFilter(imageDimension, filterLength);
    end
end

function [filteredImage, filteredBlueFT] = applyFilter(redFT, greenFT, blueFT, filterMask)
    % Apply filter to the Fourier transformed image
    filteredRedFT = redFT .* filterMask;
    filteredGreenFT = greenFT .* filterMask;
    filteredBlueFT = blueFT .* filterMask;
    
    filteredRed = ifft2(filteredRedFT);
    filteredGreen = ifft2(filteredGreenFT);
    filteredBlue = ifft2(filteredBlueFT);
    
    filteredImage = zeros(size(redFT, 1), size(redFT, 2), 3);
    filteredImage(:,:,1) = abs(filteredRed);
    filteredImage(:,:,2) = abs(filteredGreen);
    filteredImage(:,:,3) = abs(filteredBlue);
end

function displayResults(filteredImage, filteredBlueFT, blueFT, filterMask)
    % Display results in figures
    figure(1)
    image(filteredImage) % filtered image
    title('Filtered Image', 'fontsize', 20)
    
    figure(2)
    mesh(abs(fftshift(filteredBlueFT))) % filtered image in FT domain
    axis([0 600 0 600 0 10^6])
    title('Magnitude of Filtered FT of Image', 'fontsize', 20)
    
    figure(3)
    mesh(abs(fftshift(blueFT - filteredBlueFT))) % amount removed by LPF freq domain
    title('Magnitude of Values Removed from FT of Image', 'fontsize', 20)
    
    figure(4)
    mesh(ifft2(fftshift(filterMask))) % filter in image domain
    axis([270 330 270 330 0 max(max(ifft2(fftshift(filterMask))))])
    title('Filter in the Image Domain', 'fontsize', 20)
    
    figure(5)
    mesh(fftshift(real(filterMask))) % filter in FT domain
    title('Filter in the Frequency Domain', 'fontsize', 20)
    
    pause
end

function displayFinalResult(filteredImage)
    % Display final result for scene 10
    figure(1)
    mesh(abs(filteredImage(:,:,1))')
    view(90, 90)
end

% Helper functions for creating filters
function filterMask = createSquareLPF(imageDimension, filterLength)
    filterShift = zeros(imageDimension);
    filterBoxMin = imageDimension/2 - filterLength;
    filterBoxMax = imageDimension/2 + filterLength;
    filterShift(filterBoxMin:filterBoxMax, filterBoxMin:filterBoxMax) = ones(2*filterLength+1, 2*filterLength+1);
    filterMask = fftshift(filterShift);
end

function filterMask = createSquareHPF(imageDimension, filterLength)
    filterShift = ones(imageDimension);
    filterBoxMin = imageDimension/2 - filterLength;
    filterBoxMax = imageDimension/2 + filterLength;
    filterShift(filterBoxMin:filterBoxMax, filterBoxMin:filterBoxMax) = zeros(2*filterLength+1, 2*filterLength+1);
    filterMask = fftshift(filterShift);
end

function filterMask = createCircularLPF(imageDimension, filterLength)
    filterShift = zeros(imageDimension);
    [X, Y] = meshgrid(1:imageDimension, 1:imageDimension);
    filterShift((X - imageDimension/2).^2 + (Y - imageDimension/2).^2 < filterLength^2) = 1;
    filterMask = fftshift(filterShift);
end

function filterMask = createCircularHPF(imageDimension, filterLength)
    filterShift = ones(imageDimension);
    [X, Y] = meshgrid(1:imageDimension, 1:imageDimension);
    filterShift((X - imageDimension/2).^2 + (Y - imageDimension/2).^2 < filterLength^2) = 0;
    filterMask = fftshift(filterShift);
end

function filterMask = createThresholdFilter(imageDimension, filterLength)
    filterShift = (abs(fftshift(fft2(zeros(imageDimension)))) > (filterLength * 10^2));
    filterMask = fftshift(filterShift);
end

main()
