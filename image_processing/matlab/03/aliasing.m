function    aliasing

n=3;

fs=100;
f=20;
t=(0:1/fs:1);
y1=sin(2*pi*t*f);
figure(1);
subplot(2,2,1); plot(y1);
subplot(2,2,2); plot(20*log10(abs(fft(y1))+eps)); axis([1 length(y1) -50 50])
title([ 'fs=' num2str(fs) ])

y2=y1(1:n:end);

subplot(2,2,3); plot(y2);
subplot(2,2,4); plot(20*log10(abs(fft(y2))+eps)); axis([1 length(y2) -50 50])
title(['fs=' num2str(fs/n)])


Kasparov=imread('Kasparov.jpg');

figure(2);
imshow(Kasparov);
figure(3);
colormap('default') ;
imagesc(20*log10(abs(fft2(Kasparov))+eps)); 
title([ 'sub sample coef=' num2str(1) ])

Kasparov2=Kasparov(1:n:end,1:n:end);

figure(4);
imshow(Kasparov2);
figure(5);
colormap('default') ;
imagesc(20*log10(abs(fft2(Kasparov2))+eps)); 
title([ 'sub sample coef=' num2str(n) ])


