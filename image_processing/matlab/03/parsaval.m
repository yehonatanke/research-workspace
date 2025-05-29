% % parsaval
close all

Cameraman=double(imread('cameraman.jpg'));
[M,N]=size(Cameraman);
ZPN=90;
ZP_Cameraman=zeros(M+ZPN,N+ZPN);
ZP_Cameraman(1:M , 1:N)=Cameraman;
 figure;imshow(uint8(Cameraman)); title('cameraman');
 figure;imshow(uint8(ZP_Cameraman)); title('Zero padding cameraman');
figure; imagesc(20*log10(abs(fft2(Cameraman)))); colorbar; title('cameraman');
figure ; imagesc(20*log10(abs(fft2(ZP_Cameraman)))); colorbar; title('Zero padding cameraman');