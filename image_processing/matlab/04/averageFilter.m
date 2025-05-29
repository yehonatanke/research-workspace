% average filter 

Lena=imread('lena.jpg');
figure(1)
imshow(Lena)
L=0.8*double(Lena)+0.2*255*rand(size(Lena));
figure(2)
imshow(uint8(L))
title ('sinal+noise')
h = fspecial('average',3)
figure(3)
freqz2(h)
colorbar
title ('average filter  ,size=3')
L1=filter2(h,L);
figure(4)
imshow(uint8(L1))
title ('average filter  ,size=3')
pause
%  h = fspecial('average',6)
h = fspecial('gaussian',6,0.8)
figure(3)
freqz2(h)
colorbar
title ('gaussian ,size=6 sigma=0.8')
L1=filter2(h,L);
figure(4)
imshow(uint8(L1))
title ('gaussian ,size=6 sigma=0.8')
%   imwrite(L1,'lenaAvg.jpg','jpg')
