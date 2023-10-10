%% 2.1 Contrast Stretching
%a
Pc = imread('mrt-train.jpg');
whos Pc;
P = rgb2gray(Pc);

%b
figure;
imshow(P);
title('Original Image');

%c
minIntensity = double(min(P(:)));
maxIntensity = double(max(P(:)));
fprintf('Minimum intensity is: %d\n', minIntensity);
fprintf('Maximum intensity is: %d\n', maxIntensity);
% Define the desired minimum and maximum intensities (0 and 255)
desiredMin = 0;
desiredMax = 255;

%d
% Apply contrast stretching formula
P2 = (double(P) - minIntensity)*255 / (maxIntensity - minIntensity);
P2 = uint8(P2);
minIntensity2 = min(P2(:));
maxIntensity2 = max(P2(:));
fprintf('Minimum intensity in stretched image is: %d\n', minIntensity2);
fprintf('Maximum intensity in stretched image is: %d\n', maxIntensity2);

%e
figure;
imshow(P2);
title('Contrast-Stretched Image');

%% 2.2 Histogram Equalization
%a
figure;
imhist(P,10);
title('Image intensity histogram of P with 10 bins');

figure;
imhist(P,256);
title('Image intensity histogram of P with 256 bins');

%b
P3 = histeq(P,255);
figure;
imhist(P3,10);
title('Image intensity histogram of P3 with 10 bins');
figure;
imhist(P3,256);
title('Image intensity histogram of P3 with 256 bins');

%c
P4 = histeq(P3,255);
figure;
imhist(P3,10);
title('Image intensity histogram of P4 with 10 bins');
figure;
imhist(P3,256);
title('Image intensity histogram of P4 with 256 bins');

% Prompt the user for input
user_input = input('Enter "y" to close all figures or any other key to continue: ', 's');
% Check the user's input
if strcmpi(user_input, 'y')
    close all;  % Close all figures
end

%% 2.3 Linear Spatial Filtering
%a
%fspecial('gaussian',hsize,sigma): returns a rotationally symmetric Gaussian lowpass filter of size hsize with standard deviation sigma.
sigma1 = 1;  % Standard deviation for the first Gaussian filter
sigma2 = 2;  % Standard deviation for the second Gaussian filter
filter_size = 5;  % Size of the filter kernel (e.g., 5x5)

% Create Gaussian filter kernels
h1 = 1 / (2 * pi * sigma1^2) * fspecial('gaussian', filter_size, sigma1);
h2 = 1 / (2 * pi * sigma2^2) * fspecial('gaussian', filter_size, sigma2);

% Normalize the filters
h1_normalized = h1 / sum(h1(:));
h2_normalized = h2 / sum(h2(:));

% Create mesh grids for x and y
[x, y] = meshgrid(1:filter_size, 1:filter_size);

% Plot the first normalized Gaussian filter
figure;
subplot(1, 2, 1);
mesh(x, y, h1_normalized);
title('Normalized Gaussian Filter 1');
xlabel('X');
ylabel('Y');
zlabel('Value');
% Plot the second normalized Gaussian filter
subplot(1, 2, 2);
mesh(x, y, h2_normalized);
title('Normalized Gaussian Filter 2');
xlabel('X');
ylabel('Y');
zlabel('Value');

%b
Pc2 = imread('lib-gn.jpg');
whos Pc2;
figure;
imshow(Pc2);
title('lib-gn.jpg');

%c
% Filter the image using the first normalized Gaussian filter
filtered_img1 = conv2(double(Pc2), h1_normalized, 'same');

% Filter the image using the second normalized Gaussian filter
filtered_img2 = conv2(double(Pc2), h2_normalized, 'same');

% Display the original and filtered images
subplot(1, 3, 1);
imshow(Pc2);
title('lib-gn.jpg');

subplot(1, 3, 2);
imshow(uint8(filtered_img1));  % Convert back to uint8 for display
title('Filtered with Gaussian Filter 1');

subplot(1, 3, 3);
imshow(uint8(filtered_img2));  % Convert back to uint8 for display
title('Filtered with Gaussian Filter 2');

%d
Pc3 = imread('lib-sp.jpg');
whos Pc3;
figure;
imshow(Pc3);
title('lib-sp.jpg');

%e
% Filter the image using the first normalized Gaussian filter
filtered_img1 = conv2(double(Pc3), h1_normalized, 'same');

% Filter the image using the second normalized Gaussian filter
filtered_img2 = conv2(double(Pc3), h2_normalized, 'same');

% Display the original and filtered images
subplot(1, 3, 1);
imshow(Pc3);
title('lib-sp.jpg');

subplot(1, 3, 2);
imshow(uint8(filtered_img1));  % Convert back to uint8 for display
title('Filtered with Gaussian Filter 1');

subplot(1, 3, 3);
imshow(uint8(filtered_img2));  % Convert back to uint8 for display
title('Filtered with Gaussian Filter 2');

%% 2.4 Median Filtering

% Median filtering for lib-gn.jpg
Pc2 = imread('lib-gn.jpg');

% Define the sizes of the median filter windows (3x3 and 5x5)
window_size1 = [3, 3];
window_size2 = [5, 5];

% Perform median filtering with a 3x3 window
filtered_img1 = medfilt2(Pc2, window_size1);

% Perform median filtering with a 5x5 window
filtered_img2 = medfilt2(Pc2, window_size2);

% Display the original image and the filtered images
figure;
subplot(1, 3, 1);
imshow(Pc2);
title('lib-gn.jpg');

subplot(1, 3, 2);
imshow(filtered_img1);
title('lib-gn.jpg Median Filtered (3x3)');

subplot(1, 3, 3);
imshow(filtered_img2);
title('lib-gn.jpg Median Filtered (5x5)');

% Median filtering for lib-sp.jpg
Pc3 = imread('lib-sp.jpg');

% Define the sizes of the median filter windows (3x3 and 5x5)
window_size1 = [3, 3];
window_size2 = [5, 5];

% Perform median filtering with a 3x3 window
filtered_img1 = medfilt2(Pc3, window_size1);

% Perform median filtering with a 5x5 window
filtered_img2 = medfilt2(Pc3, window_size2);

% Display the original image and the filtered images
figure;
subplot(1, 3, 1);
imshow(Pc3);
title('lib-sp.jpg');

subplot(1, 3, 2);
imshow(filtered_img1);
title('lib-sp.jpg Median Filtered (3x3)');

subplot(1, 3, 3);
imshow(filtered_img2);
title('lib-sp.jpg Median Filtered (5x5)');

%% 2.5 Suppressing Noise Interference Patterns 
%a
Pc4 = imread('pck-int.jpg');
whos Pc4;
figure;
imshow(Pc4);
title('pck-int.jpg');

%b
% Compute the Fourier transform of the image
F = fft2(double(Pc4));
S = abs(F);
figure;
imagesc(fftshift(S.^0.1));
title('Power Spectrum with fftshift');
colormap('default');

%c
figure;
imagesc(S.^0.1);
title('Power Spectrum without fftshift');
colormap('default');

x1 = 249;
y1 = 17;
x2 = 9;
y2 = 241;

%d 
F(y1-2 : y1+2, x1-2 : x1+2) = 0;
F(y2-2 : y2+2, x2-2 : x2+2) = 0;
S = abs(F);
figure;
imagesc(fftshift(S.^0.1));
title('Power Spectrum with fftshift after making peaks 0');
colormap('default');

%e
img = uint8(ifft2(F));
figure;
imshow(img)
title('Inverse Fast Fourier Transform (IFFT)');

% Additional Filtering
F(y1, :) = 0;
F(y2, :) = 0;
F(:, x1) = 0;
F(:, x2) = 0;
S = abs(F);
figure;
imagesc(fftshift(S.^0.1));
title('additional filtering');
img = uint8(ifft2(F));
figure;
imshow(img);
title('IFFT after additional filtering');

% Contrast Stretching
r_min = double(min(img(:)));
r_max = double(max(img(:)));
whos img
img = uint8(255 * (double(img) - r_min) / (r_max - r_min));
whos img;
figure;
imshow(img);
title('Contrast stretching after additional filtering');

%f
Pc5 = imread('primate-caged.jpg');
whos Pc5;
Pc5 = rgb2gray(Pc5);
figure;
imshow(Pc5);
title('primate-caged.jpg');

F = fft2(Pc5);
S = abs(F);
figure;
imagesc(fftshift(S.^0.1));
title('Power spectrum of primate-caged.jpg with fftshift');

x1 = 11;
y1 = 252;
x2 = 247;
y2 = 6;
x3 = 21;
y3 = 248;
x4 = 237;
y4 = 10;
F(y1-2 : y1+2, x1-2 : x1+2) = 0;
F(y2-2 : y2+2, x2-2 : x2+2) = 0;
F(y3-2 : y3+2, x3-2 : x3+2) = 0;
F(y4-2 : y4+2, x4-2 : x4+2) = 0;
S = abs(F);
figure;
imagesc(fftshift(S.^0.1));

% Display new image
img = uint8(ifft2(F));
figure;
imshow(img)
title('Final Result');
% Further performing contrast filtering does not significantly improve the result.

%% 2.6 Undoing Perspective Distortion of Planar Surface
%a
Pc6 = imread('book.jpg');
whos Pc6;
figure;
imshow(Pc6);
title('book.jpg');

%b
% [X Y] = ginput(4);
% For replicating experiment
 X = [143.0000 309.0000 7.0000 255.0000];
 Y = [28.0000 48.0000 160.0000 213.0000];
imageX = [0 210 0 210];
imageY = [0 0 297 297];

%c
A = [
	[X(1), Y(1), 1, 0, 0, 0, -imageX(1)*X(1), -imageX(1)*Y(1)];
	[0, 0, 0, X(1), Y(1), 1, -imageY(1)*X(1), -imageY(1)*Y(1)];
	[X(2), Y(2), 1, 0, 0, 0, -imageX(2)*X(2), -imageX(2)*Y(2)];
	[0, 0, 0, X(2), Y(2), 1, -imageY(2)*X(2), -imageY(2)*Y(2)];
	[X(3), Y(3), 1, 0, 0, 0, -imageX(3)*X(3), -imageX(3)*Y(3)];
	[0, 0, 0, X(3), Y(3), 1, -imageY(3)*X(3), -imageY(3)*Y(3)];
	[X(4), Y(4), 1, 0, 0, 0, -imageX(4)*X(4), -imageX(4)*Y(4)];
	[0, 0, 0, X(4), Y(4), 1, -imageY(4)*X(4), -imageY(4)*Y(4)];
];
v = [imageX(1); imageY(1); imageX(2); imageY(2); imageX(3); imageY(3); imageX(4); imageY(4)];

u = A \ v;
U = reshape([u;1], 3, 3); 
w = U*[X; Y; ones(1,4)];
w = w ./ (ones(3,1) * w(3,:));

%d
T = maketform('projective', U);
P2 = imtransform(Pc6, T, 'XData', [0 210], 'YData', [0 297]);

%e
figure;
imshow(P2);
%imwrite(P2, "straightbook.jpg");

%f
whos P2;
% Define the target color range for pink
pinkLowerBound = [151, 90, 62];
pinkUpperBound = [255, 170, 154];

% Create a binary mask
binaryMask = (P2(:,:,1) >= pinkLowerBound(1) & P2(:,:,1) <= pinkUpperBound(1)) ...
             & (P2(:,:,2) >= pinkLowerBound(2) & P2(:,:,2) <= pinkUpperBound(2)) ...
             & (P2(:,:,3) >= pinkLowerBound(3) & P2(:,:,3) <= pinkUpperBound(3));

% Visualize the identified pink area
maskedImage = P2;
maskedImage(repmat(~binaryMask, [1 1 3])) = 0; % Set non-pink pixels to black

figure;
imshow(maskedImage);
title('Identified Pink Area');

% Label connected components in the binary mask
pinkLabelMatrix = bwlabel(binaryMask);

% Analyze the properties of connected components
pinkStats = regionprops(pinkLabelMatrix, 'Area');

% Define a threshold for the minimum area (adjust as needed)
minPinkAreaThreshold = 50;

% Create a binary mask to keep only large pink regions
largePinkBinaryMask = ismember(pinkLabelMatrix, find([pinkStats.Area] >= minPinkAreaThreshold));

% Filter the maskedImage to keep only large pink regions
largePinkImage = maskedImage;
largePinkImage(repmat(~largePinkBinaryMask, [1 1 3])) = 0; % Set non-large pink pixels to black

% Display the result
figure;
imshow(largePinkImage);
title('Large Pink Areas');

% Perform morphological operations
se = strel('rectangle', [6, 6]); % Define a rectangular structuring element (adjust the size as needed)

% Erosion to remove small artifacts (optional)
erodedMask = imerode(binaryMask, se);

% Dilation to expand the pink region back to the original size
dilatedMask = imdilate(erodedMask, se);

% Use the dilated mask to keep only the pink regions
resultImage = P2;
resultImage(repmat(~dilatedMask, [1 1 3])) = 0; % Set non-pink pixels to black

% Display the result
figure;
imshow(resultImage);
title('Refined Pink Areas - Morphological operations');



