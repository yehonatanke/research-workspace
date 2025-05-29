function [ImageOut]=random_dithering(ImageIn)

if (nargin<1)
    DialogTitle='Input file name';
    FilterSpec={'*.jpg'; '*.png' ; '*.tif' ;'*.gif'};
    DefaultName='lena.gif';
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

 if(ndims(ImageIn)==3)
     ImageIn=rgb2gray(ImageIn);
 end
close all  % figure; %
[high,width]=size(ImageIn);
c_h=ceil(high/2);  Chigh=2*c_h;
c_w=ceil(width/2) ; Cwidth=2*c_w;
ImagePad=zeros(Chigh,Cwidth);
ImagePad(1:high,1:width)=uint8(ImageIn);

A=[51 153 204 102];
B=perms(1:4);           % all the permutations
Lb=factorial(4);        % 4!
R=randi(Lb,c_h,c_w);    % permutation indexes matrix
DithMat=zeros(Chigh,Cwidth);
for k=0:c_h-1
    for j=0:c_w-1
        pv=A(B(R(k+1,j+1),:));
        C=reshape(pv,2,2);
        DithMat(2*k+1:2*k+2,2*j+1:2*j+2)=C;
    end
end

ImagePad=(ImagePad>DithMat);
ImageOut=ImagePad(1:high,1:width);
ImageOut=uint8(255*ImageOut);

if (nargout<1)
    montage ([ImageIn ImageOut dithering(ImageIn)]);
    title('randon\_dithering');
    clear ImageOut
end






