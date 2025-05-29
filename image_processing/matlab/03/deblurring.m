% function deblurring(Im)
%%  initialization
close all
clear all

% tmp=nargin;
tmp=0;
if( tmp<1)
[FileName,PathName,FilterIndex] = uigetfile('*.jpg','Please select input Image');
if (FilterIndex==0)
    disp('No input image was selected.');
    return
end

PicName=strcat(PathName,FileName);
Im=imread(PicName);
end

%% 
%h = fspecial('gaussian', hsize, sigma)
BlurKernel=fspecial('gaussian', [13 13],0.7);
figure; imagesc(log10(BlurKernel+eps)); colorbar;
title('convolution kernel');

%%
figure; imshow(Im); title('Original Image');
BlurIm=imfilter(Im,BlurKernel,'conv','same','symmetric');
figure; imshow(BlurIm); title('Blurred Image');
%%
SzBlurIm=size(BlurIm);
SzBlurKernel=size(BlurKernel);
CmSize=SzBlurIm+ SzBlurKernel -[1 1];

%B = PADARRAY(A,PADSIZE,PADVAL) 
PdBlurIm=padarray(BlurIm,CmSize-SzBlurIm,'post' );
PdBlurKernel=padarray(BlurKernel,CmSize-SzBlurKernel,'post' );
figure; imshow(PdBlurIm); title('Blurred Image after Zero padding');
%%  Restoration
XBlurIm=fft2(PdBlurIm);
XBlurKernel=fft2(PdBlurKernel);

XRestorIm=XBlurIm./(XBlurKernel+eps);

RestorPdIm=uint8(abs(ifft2(XRestorIm)));
figure; imshow(RestorPdIm); title('Restored zero padded Image');

RestorIm=RestorPdIm(1:SzBlurIm(1),1:SzBlurIm(2));
figure; imshow(RestorIm); title('Restored  Image');
%%
picarray=[Im BlurIm RestorIm];
close all
montage(picarray);
