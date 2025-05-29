function  block_process

close all
A=imread('book_copying.tif');
S=size(A); m=S(1); n=S(2);
N=20;
epsilon=0.01;
m=fix(m/N); n=fix(n/N); m=m*N; n=n*N;
B=A(1:m,1:n);
figure; imshow(B,[]);
fun = @mean2;
bg_mean=blkproc(B,[N N],fun);
bg_bkgd=imresize(bg_mean,size(B),'bilinear');
gain=max(max(im2double(bg_bkgd)))./(im2double(bg_bkgd)+epsilon);
restor_image=im2double(B).*gain;
restor_image=restor_image/max(max(restor_image));

L=[ linspace(0,1,200)   ones(1,256-200)];
B=fix(im2double(restor_image)*255+1); 
C=L(B);
figure; imshow((C));  title('processed image');
