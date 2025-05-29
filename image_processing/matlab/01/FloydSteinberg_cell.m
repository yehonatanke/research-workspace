%function []=FloydSteinberg_cell


%%  first cell reading an image

   InImFname='lena.gif';

InIm=imread(InImFname);


%%   second cell showing original picture

[height,width]=size(InIm); 
figure('Position',[1,512,width,height]);
imshow(InIm);
title('original picture');

%%  third cell : processing
OutIm=error_diffusion(InIm);
[height,width]=size(OutIm);  figure('Position',[512,512,width,height]); colormap(gray(256));
imshow(OutIm); title('error diffusion result');










