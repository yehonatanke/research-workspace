function [ImageOut]=dithering(ImageIn)

if (nargin<1)
    DialogTitle='Input file name';
    FilterSpec={'*.jpg'; '*.png' ; '*.tif' ;'*.gif'};
    DefaultName='lena.gif';
    [FileName,PathName,FilterIndex] =...
        uigetfile(FilterSpec,DialogTitle,DefaultName) ;
    if ( isequal(FilterIndex,0) )
   disp('User selected Cancel')
   return;
    else
    ImageFullName=fullfile(PathName, FileName);
   disp(['User selected:  ', ImageFullName]);
    ImageIn=imread(ImageFullName);
    end
end

close all  % figure; %
[high,width,depth]=size(ImageIn);
c_h=ceil(high/2);  Chigh=2*c_h;
c_w=ceil(width/2) ; Cwidth=2*c_w;
ImageTmp=zeros(Chigh,Cwidth);
if (depth==3) %(ndims(ImageIn)==3) % color
    ImageIn=rgb2gray(ImageIn);
end
ImageTmp(1:high,1:width)=uint8(ImageIn);
DithMat=repmat([51 153;204 102],c_h,c_w);
ImageTemp=(ImageTmp>DithMat);
ImageOut=ImageTemp(1:high,1:width);
ImageOut=uint8(255*ImageOut);


if (nargout<1)
    montage ([ImageIn ImageOut]);
    title('dithering');
    clear ImageOut
end






