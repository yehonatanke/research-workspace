function ImageComposition
%  initialization
close all
clear all
epsilon=5e3;
lepsilon=20*log10(epsilon);
lBepsilon=20*log10(1e-3);
if (nargin<1)
    DialogTitle='Input file name';
    FilterSpec={'*.jpg'; '*.png' ; '*.tif' ;'*.gif'};
    DefaultName='jap128.jpg';
    [FileName,PathName,FilterIndex] =uigetfile(FilterSpec,DialogTitle,DefaultName) ;
    if ( isequal(FilterIndex,0) )
        disp('User selected Cancel')
        return;
    else
        ImageFullName=fullfile(PathName, FileName);
        disp(['User selected:  ', ImageFullName]);
        ImageIn=imread(ImageFullName);
    end
end
figure(1); imshow(ImageIn); title('Original Image');
[M,N]=size(ImageIn);
FrImageIm=fft2(ImageIn);
A=FrImageIm;
B=20*log10(abs(A)+eps);
figure(2); imagesc(20*log10(B+eps)); title('Fourier Coefficients');
colorbar;
Gmask=zeros(M,N);
count=0;
[C,v1]=max(B,[],2);
[D,m]=max(C);
n=v1(m);
figure(3);
while( D>lepsilon)
    mask=zeros(M,N);
    mask(m,n)=1;
    mask(pairs(m,n,M,N))=1;
    Gmask(m,n)=1;
    Gmask(pairs(m,n,M,N))=1;
    F=mask.*FrImageIm;
    B(mask==1)=lBepsilon;
    Im=real(ifft2(F));
    G=Gmask.*FrImageIm;
    GIm=real(ifft2(G));
    if (mod(count,200)==0)
        Sim=scaleIm(Im);
        montage([Sim scaleIm(GIm) scaleIm(B) ]);
        title(strcat(num2str(m),',',num2str(n)));
        pause(0.02)
    end
    count=count+1;
    % find the indexes of the current maxima
    [C,v1]=max(B,[],2);
    [D,m]=max(C);
    n=v1(m);    
end
figure(4);  montage([ImageIn GIm]) ;

function [m1,n1]=pairs(m,n,M,N)

m1=M-m+2;
n1=N-n+2;
if(m1>M)
    m1=1;
end
if(n1>N)
    n1=1;
end

function Im=scaleIm(Im)
m=min(min(Im));
M=max(max(Im));
if(M~=m)
    Im=uint8(255*(Im-m)/(M-m));
end
