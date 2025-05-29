%function    Furier_analysis

%%  Image import
A=imread('lena','gif');
close all;
figure(1); imshow(A);

%%  image disassembly in to frequency basis 
FF=fft2(A);
[M,N]=size(FF);
figure(2);imagesc(20*log(abs(FF)+1)); title('DFT coefficients'); colorbar;
colormap('default');

mask=zeros(M,N);  m=1; n=1; mask(m,n)=1;
GF=mask.*FF;
B=ifft2(GF);
disp(strcat(' max imag is :',num2str(max(max(imag(B))))));
figure(3); imagesc(20*log(abs(B)+1));title('m=1 n=1');  colormap('default'); colorbar;



mask=zeros(M,N);  m=4; n=3; sh=2; mask(m,n)=1; mask(M-m+sh,n)=1; mask(m,N-n+sh)=1; mask(M-m+sh,N-n+sh)=1; 
GF=mask.*FF;
B=ifft2(GF);
disp(strcat(' max imag is :',num2str(max(max(imag(B))))));
figure(4); imagesc(20*log(abs(B)+1)); colormap('default'); title('m=3 n=4');  colorbar;


mask=zeros(M,N);  m=144; n=71; sh=2; mask(m,n)=1; mask(M-m+sh,n)=1; mask(m,N-n+sh)=1; mask(M-m+sh,N-n+sh)=1; 
GF=mask.*FF;
B=ifft2(GF);
disp(strcat(' max imag is :',num2str(max(max(imag(B))))));
figure(5); imagesc(20*log(abs(B)+1));title('m=144 n=71');colormap('default');  colorbar;



%%  Low frequencies image content 

A=imread('lena','gif');
FF=fft2(A);
[M,N]=size(FF);
mask=zeros(M,N);
for k=2:floor(M/4)
    mask(k,1)=1;
    for j=2:floor(N/4)
        mask(1,j)=1;
        m=k; n=j; sh=2; mask(m,n)=1; mask(M-m+sh,n)=1; mask(m,N-n+sh)=1; mask(M-m+sh,N-n+sh)=1;
    end
end

GF=mask.*FF; C1=real(ifft2(GF)) ;  B=uint8(C1);
figure(6); imshow(B); title('low frequency content');


%%  High  frequencies image content

mask=1-mask;
GF=mask.*FF;   C2=real(ifft2(GF)) ;  B=uint8(C2);
figure(7); imshow(B); title('high frequency content');


%%  High and low frequency assembly

 B=uint8(C1+C2);
figure(8); imshow(B); title('Image assembly');