Lena=imread('lena.gif');
figure(1)
imshow(Lena);
ind=find(rand(size(Lena))>0.99);
L=double(Lena);
L(ind)=0;
figure(2)
imshow(uint8(L))
L1=medfilt2(L,[5 5]);
figure(3)
imshow(uint8(L1))
