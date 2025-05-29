function m_laplacian
alpha =0.2;
h1 = fspecial('laplacian',alpha ) ;
A=imread('moon.tif');
figure(1) ; imshow(A)
A1=filter2(h1,A);
A2=double(A) - 0.6*double(A1);
figure(2);imshow(uint8(A2)); title('unsharp')