%function    segmentation

%%  Image import
A=imread('tia','jpg');
figure(1); imshow(A);

%%  Region of interest definition

msg='   Use mouse to point ROI ';
disp( msg);    h4=msgbox(msg) ; position=get(h4,'position');
set(h4,'position',position.*[0.5 0.5 1 1]);
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
figure(4); scatter3(J(:,1),J(:,2),J(:,3),10,J/256);

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
figure(5);  imshow(g);

