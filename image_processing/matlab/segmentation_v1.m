%function    segmentation

%%  Image import
A=imread('tia','bmp');
figure(1); imshow(A);

%%  Region of interest definition

mask=roipoly(A);
figure(2); imshow(mask);
% Just to show you around
red=immultiply(mask,A(:,:,1));
green=immultiply(mask,A(:,:,2));
blue=immultiply(mask,A(:,:,3));
g=cat(3, red, green, blue);
figure(3);  imshow(g);

%%  Covariance  matrix and the  Mean vector estimation
ind=find(mask);
 [M,N,K]=size(A);
I=reshape(A,M*N, K);
J=double(I(ind,1:3));
 m=mean(J);
C=cov(J);

%%  segmentation  by color distance

J=double(I( : ,1:3));
D=repmat(m,M*N,1);
D=J-D;
epsilon=100;
mask=uint8(reshape( (  sum(  ( D*inv(C) ).*D,2)  < epsilon ),M,N));
red=immultiply(mask,A(:,:,1));
green=immultiply(mask,A(:,:,2));
blue=immultiply(mask,A(:,:,3));
g=cat(3, red, green, blue);
figure(4);  imshow(g);

