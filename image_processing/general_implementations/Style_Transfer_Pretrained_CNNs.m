% Style Transfer using Pretrained VGG-19 CNN 

% Load content and style images
content_img = imread('content_image.jpg');
style_img = imread('style_image.jpg');

% Resize images to the same dimensions
image_size = [400, 400];
content_img = imresize(content_img, image_size);
style_img = imresize(style_img, image_size);

% Display content and style images
figure, imshow(content_img);
title('Content Image');
figure, imshow(style_img);
title('Style Image');

% Load pre-trained VGG-19 network
net = vgg19;

% Extract content and style layer names for VGG-19
content_layer = 'conv4_2'; % Layer for content representation
style_layers = {'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'}; % Layers for style

% Extract feature maps for content and style images
content_features = activations(net, content_img, content_layer, 'OutputAs', 'channels');
style_features = cell(1, numel(style_layers));
for i = 1:numel(style_layers)
    style_features{i} = activations(net, style_img, style_layers{i}, 'OutputAs', 'channels');
end

% Initialize the stylized image (start with the content image)
stylized_img = content_img;

% Define weights for content and style losses
content_weight = 1e3;
style_weight = 1e-2;
learning_rate = 2;
num_iterations = 500;

% Optimization loop to minimize the content and style losses
for iter = 1:num_iterations
    % Extract feature maps of the stylized image
    stylized_content_features = activations(net, stylized_img, content_layer, 'OutputAs', 'channels');
    
    % Compute content loss (mean squared error)
    content_loss = content_weight * sum((stylized_content_features - content_features).^2, 'all');
    
    % Compute style loss using Gram matrices
    style_loss = 0;
    for i = 1:numel(style_layers)
        stylized_style_features = activations(net, stylized_img, style_layers{i}, 'OutputAs', 'channels');
        style_loss = style_loss + style_weight * computeStyleLoss(stylized_style_features, style_features{i});
    end
    
    % Compute total loss (content + style)
    total_loss = content_loss + style_loss;
    
    % Update the stylized image using gradient descent
    grad = gradientOfImage(total_loss, stylized_img); % Custom function for gradient calculation
    stylized_img = stylized_img - learning_rate * grad;
    
    % Display progress
    if mod(iter, 50) == 0
        disp(['Iteration: ', num2str(iter), ', Total Loss: ', num2str(total_loss)]);
        imshow(uint8(stylized_img)); % Show intermediate results
        drawnow;
    end
end

% Display the final stylized image
figure, imshow(uint8(stylized_img));
title('Final Stylized Image');

% Save the stylized image
imwrite(uint8(stylized_img), 'stylized_image.jpg');


% Helper function to compute style loss (Gram matrix)
function loss = computeStyleLoss(stylized_features, style_features)
    gram_stylized = gramMatrix(stylized_features);
    gram_style = gramMatrix(style_features);
    loss = sum((gram_stylized - gram_style).^2, 'all');
end

% Helper function to compute Gram matrix
function G = gramMatrix(features)
    [H, W, C] = size(features);
    reshaped_features = reshape(features, H*W, C);
    G = reshaped_features' * reshaped_features / (H * W * C);
end

% Custom function to compute the gradient (simplified for clarity)
function grad = gradientOfImage(loss, img)
    grad = img * 0.01; % Placeholder gradient for demonstration (use actual derivative)
end
